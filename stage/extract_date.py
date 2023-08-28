import pytesseract
import re
from PIL import Image

# Chemin vers l'exécutable Tesseract (modifiez-le en fonction de votre configuration)

def extract_dates_from_image(image_path):
    try:
        # Charger l'image
        image = Image.open(image_path)

        # Utiliser Tesseract pour effectuer l'OCR sur l'image
        ocr_text = pytesseract.image_to_string(image)

        # Utiliser une expression régulière pour trouver les dates dans le texte OCR
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        dates_found = re.findall(date_pattern, ocr_text)

        if dates_found:
            # Convertir les dates trouvées en objets de date pour pouvoir les comparer
            parsed_dates = [date.replace('/', '-') for date in dates_found]

            # Trouver la date la plus récente
            most_recent_date = max(parsed_dates)

            return most_recent_date
        else:
            return "Aucune date trouvée dans l'image."

    except Exception as e:
        return str(e)

# Chemin vers l'image à traiter (modifiez-le en fonction de votre image)
image_path = 'cheqq.jpg'
most_recent_date = extract_dates_from_image(image_path)
print("Date la plus récente trouvée :", most_recent_date)
