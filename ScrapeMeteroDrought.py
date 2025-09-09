# code to scrape data from sissa website for metero drought
# now its working first remove dawnload dir and then create
# if csv contains more than 3/4th values missing then remove from directory else kept and fill with ffill (27 outoff 36 missing then remove)
# aded popup cancel button to click
# Final code (takes 70-80 min to dawnload all valis csv 76 outof 138)
## code for cloud function

import os
import time
import pandas as pd
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import glob
import shutil
from google.cloud import storage
from flask import Flask, request, jsonify

def clean_and_rename_csv(download_dir, city_name):
    # Find the "grafico series temporalesserie t.csv" file
    target_pattern = os.path.join(download_dir, "grafico series temporalesserie t.csv")
    city_csv = os.path.join(download_dir, f"{city_name}.csv")
    found = False

    # Rename the target file to citywise name
    if os.path.exists(target_pattern):
        os.rename(target_pattern, city_csv)
        found = True

    # Remove ONLY the temp file if it still exists (shouldn't, but just in case)
    for csv_file in glob.glob(os.path.join(download_dir, "*.csv")):
        if os.path.basename(csv_file) == "grafico series temporales boxplot.csv":
            os.remove(csv_file)

    if found:
        print(f"Saved: {city_csv}")
    else:
        print("No 'grafico series temporalesserie t.csv' found to rename.")

def wait_and_click(driver, by, value, timeout=20):
    """Wait for and click an element with retries and overlay handling, with robust scrolling."""
    wait = WebDriverWait(driver, timeout)
    for _ in range(3):
        try:
            element = wait.until(EC.element_to_be_clickable((by, value)))
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element
            )
            time.sleep(1)
            driver.execute_script("arguments[0].click();", element)
            time.sleep(2)
            return True
        except Exception:
            try:
                ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(1)
            except:
                pass
    return False

def close_cerrar_popup(driver):
    """Detect and click the 'Cerrar' button in popup if present."""
    try:
        cerrar_btn = driver.find_element(By.XPATH, "//button[span[text()='Cerrar']]")
        if cerrar_btn.is_displayed():
            cerrar_btn.click()
            time.sleep(1)
            print("Popup closed by clicking 'Cerrar'.")
            return True
    except Exception:
        pass
    return False

def wait_for_download(download_dir, timeout=20):
    """Wait for a file to be downloaded"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        files = os.listdir(download_dir)
        if any(f.endswith('.csv') for f in files):
            time.sleep(2)
            return True
        time.sleep(1)
    return False

def verify_csv_file(filepath):  ###keep files having valid and not more that 27 nulls outoff 36
    """Keep file only if 'valor_indice' column exists and number of nulls < 9.
    If there are nulls, fill them with forward fill (ffill)"""
    try:
        df = pd.read_csv(filepath)
        if 'valor_indice' not in df.columns:
            return False
        null_count = df['valor_indice'].isnull().sum()
        if null_count < 9:
            if null_count > 0:
                df['valor_indice'] = df['valor_indice'].ffill()
                df.to_csv(filepath, index=False)
            return True
        else:
            return False
    except Exception:
        return False

def initialize_driver(download_dir):
    """Initialize and return a new WebDriver instance"""
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--headless=new")  # <---for hiding chrome window

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """
    })
    return driver

def process_default_city(driver, wait, download_dir):
    """Process all cities/stations and download their CSVs (renamed) in batches of 15."""
    try:
        # Set Chrome to allow multiple automatic downloads
        driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {
                "behavior": "allow",
                "downloadPath": download_dir
            }
        )

        # Open station dropdown and get all city/station options
        wait_and_click(driver, By.ID, "mat-select-2")
        options = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//mat-option")))
        total_cities = len(options)
        batch_size = 15

        print(f"Number of cities/stations in the dropdown: {total_cities}")

        for batch_start in range(1, total_cities + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_cities)
            print(f"\nProcessing cities {batch_start} to {batch_end}")

            for idx in range(batch_start, batch_end + 1):
                # Open dropdown each time (it closes after selection)
                wait_and_click(driver, By.ID, "mat-select-2")
                options = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//mat-option")))
                city_name = options[idx - 1].text.strip().replace("/", "_").replace("\\", "_")
                print(f"Selecting city/station: {city_name}")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", options[idx - 1])
                time.sleep(1)
                driver.execute_script("arguments[0].click();", options[idx - 1])
                time.sleep(2)

                # print("Selecting indice...")
                # wait_and_click(driver, By.ID, "mat-select-4")
                # wait_and_click(driver, By.XPATH, "//mat-option[1]")
                # time.sleep(1)

                print("Selecting Escala temporal...")
                wait_and_click(driver, By.ID, "mat-select-6")
                wait_and_click(driver, By.XPATH, "//mat-option[10]")
                time.sleep(1)

                print("Selecting Periodicidad de datos...")
                wait_and_click(driver, By.XPATH, "//span[contains(text(), 'Meses (6 péntadas)')]")
                time.sleep(1)

                print("Selecting Cantidad de péntadas a mostrar...")
                wait_and_click(driver, By.XPATH, "//span[@class='mat-radio-label-content' and contains(., '36')]")
                time.sleep(1)

                # Click "Visualizar"
                print("Selecting visualizer...")
                wait_and_click(driver, By.XPATH, "//button[contains(., 'Visualizar')]")
                time.sleep(5)
                close_cerrar_popup(driver)

                try:
                    no_data = driver.find_element(By.XPATH, "//*[contains(text(), 'No hay datos disponibles')]")
                    if no_data.is_displayed():
                        print(f"No data available for {city_name}. Skipping.")
                        continue
                except NoSuchElementException:
                    pass

                # Download CSV
                print(f"Downloading CSV for {city_name}...")
                wait_and_click(driver, By.XPATH, "//button[contains(., 'Descargar CSV')]")
                close_cerrar_popup(driver)
                if not wait_for_download(download_dir):
                    print(f"Download timeout - no CSV file received for {city_name}")
                    continue

                # Clean up and rename only the correct CSV
                clean_and_rename_csv(download_dir, city_name)
                city_csv = os.path.join(download_dir, f"{city_name}.csv")

                if not verify_csv_file(city_csv):
                    print(f"Downloaded CSV file has more than half null valor_indice for {city_name}, removing.")
                    os.remove(city_csv)
                    continue

                print(f"Successfully downloaded CSV for {city_name}")

            # After each batch except the last, sleep and refresh
            if batch_end < total_cities:
                print("Sleeping before next batch...")
                time.sleep(10)  # Adjust sleep time as needed
                driver.refresh()
                time.sleep(10)  # Wait for page to reload

        return True

    except Exception as e:
        print(f"Error processing default city/station: {e}")
        return False

def metero_run_sissa_download():
    """Run the SISSA download process for all cities/stations, keeping only citywise CSVs."""
    # Setup paths
    current_dir = Path(__file__).resolve().parent
    download_dir = str(current_dir / "downloads_metero")
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    # DO NOT REMOVE any existing CSVs at the start
    driver = None
    try:
        driver = initialize_driver(download_dir)
        wait = WebDriverWait(driver, 20)

        print("Opening SISSA website...")
        driver.get("https://dashboard.crc-sas.org/sissa/indices-de-sequia/series-temporales")
        time.sleep(15)

        # Process all cities/stations
        print("\nProcessing all cities/stations...")
        if process_default_city(driver, wait, download_dir):
            print("Meterological Droughts Download complete.")
        else:
            print("Failed to download Meterological Droughts data for the cities/stations.")

    except Exception as e:
        print(f"\nError occurred: {e}")
        if driver:
            driver.save_screenshot(str(current_dir / "error_screenshot.png"))
    finally:
        if driver:
            driver.quit()
            
        # --- Cleanup step: keep only city CSVs, remove others ---                       
        download_dir = str(Path(__file__).resolve().parent / "downloads_metero")
        # Get all city CSVs (already renamed)
        city_csvs = set()
        for fname in os.listdir(download_dir):
            if fname.endswith(".csv"):
                city_csvs.add(fname)
    
# --- Flask app for Cloud Run HTTP trigger ---
app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def run_metero_scraper():
    try:
        metero_run_sissa_download()
        # Optionally, insert data into your database here
        return jsonify({"status": "success", "message": "Scraping complete."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # For local testing or Cloud Run
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))