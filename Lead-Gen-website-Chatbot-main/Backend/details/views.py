import os
import smtplib
from airtable import Airtable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from .model import Details

load_dotenv()

AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY')
AIRTABLE_BASE_KEY = os.getenv('AIRTABLE_BASE_KEY')
AIRTABLE_TABLE_NAME = os.getenv('AIRTABLE_TABLE_NAME')

SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = os.getenv('SMTP_PORT')
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
TO_EMAIL = os.getenv('TO_EMAIL')

airtable = Airtable(
    AIRTABLE_BASE_KEY,
    AIRTABLE_API_KEY
    )

def send_email(lead_details):
    try:
        # Create the email message
        subject = "New Lead Details"
        body = (
            "You have received new lead details:\n\n"
            f"Name: {lead_details['name']}\n"
            f"Email: {lead_details['email']}\n"
            f"Company: {lead_details['company']}\n"
            f"Requirements: {lead_details['requirements']}\n"
        )

        message = MIMEMultipart()
        message['From'] = SMTP_USER
        message['To'] = TO_EMAIL
        message['Subject'] = subject

        # Attach the email body
        message.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, TO_EMAIL, message.as_string())

        return 'Email sent successfully'
    except Exception as e:
        return e

async def get_details(_details: Details):
    try:
        lead_details = {
            'name': _details.name,
            'email': _details.email,
            'company': _details.company,
            'requirements': _details.requirements,
        }

        airtable.create(AIRTABLE_TABLE_NAME, lead_details)
    
        status = send_email(lead_details)

        return {'message': status}
    except Exception as e:
        return {'error': str(e)}

