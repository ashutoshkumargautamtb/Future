# FUTURE AI - THE LAST PROJECT

## Modules

### SMTP
- API Sending (custom)
- Random Generator

### Dev 2
#### Content Gen Module
- Initial (HTML) => [text, pdf, image]

#### API Handler Module
- Best GEN AI APIs

#### Custom-Trained-Model
- APIs by (Services Marketplace -> Google, IBM, etc.)

#### Customer Email Module
- Classifier data verifier and cleaning -> export

#### Email Sender Module
- sendEmail.js

#### Subject Module

#### Sender Name Module

#### Self Testing Module
- Inboxing

#### Self Update Fetching Module

#### Self Encryption Module
- Delete (by itself in case of emergency)

#### Self Feedback Loop
- Call verify whether response is received or not by using API (Vonage, 8x8, RingOneCore, etc. use one only. new*)

#### Human Feedback Loop
- Take feedback from human (inboxing, subject, sender name, content, etc.)

#### Email Tracking Module
- Track location, track open ratio, etc.

#### Internet Access
- Fine-tuned on only email services updates from Google forums, etc.

#### Coder
- Code and execute and self-debug (testing, and if success then in production)

#### AI Health Module
- Overall health

#### AI API Module
- Access with token

#### AI Interface
- Web (NO CLI, NO Desktop APP) only Web app (2 interfaces: normal and developer). (LOGIN ONLY. no registration.)

#### Docker Service
- In case of emergency, transport the AI to another hidden place or machine (AWS, it could be anything anywhere)

#### One Click Setup

### Dev 1
#### Database Part
- Database 1 (DB1) Stores all data (excluding AI training data)
- Database 2 (DB2) Stores the main AI training data

### New Terms
- PSW (Potential Spammed Words)

### DB1: Headers
| ID | Email               | Content (TXT) | Content (HTML)     | PDF-ID                                      | Email-Header                                | Classification | PSW | Sending Method |
|----|---------------------|---------------|--------------------|---------------------------------------------|---------------------------------------------|----------------|-----|----------------|
| 1  | example@domain.com  | Hi! There     | <p>Hi! There</p>   | https://aws.amazon/hdhhsb/inv-836374.pdf    | Delivered-To: lucifermorningstar25456@gmail.com | INB            | na  | SMTP/API       |


Date :- 15/02/2025
________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# contentGEN.js dependency
html-pdf-node
puppeteer
pdf-lib

Subprocess Command :- node script.js <action> <htmlSource> <outputPath>
# actions
html-to-pdf
html-to-image
html-to-imgpdf

# Example Code of Python For Subprocess Commands for contentGEN.js

import subprocess
def execute_node_script(action, html_source, output_path):
    """
    Executes the Node.js script with the given parameters.
    :param action: Action to perform (html-to-pdf, html-to-image, html-to-imgpdf)
    :param html_source: Path to HTML file or HTML source code
    :param output_path: Path to save the output file
    """
    try:
        result = subprocess.run(
            ['node', 'script.js', action, html_source, output_path],
            text=True,
            capture_output=True
        )
        print(result.stdout)
        if result.stderr:
            print('Error:', result.stderr)
    except Exception as e:
        print(f"Failed to execute Node.js script: {e}")

# Example usage
html_file_path = 'D:/Workspace/Work/INV-8938-89.html'  # Path to HTML file
output_pdf_path = 'output.pdf'  # Path for PDF
output_image_path = 'output.png'  # Path for image
output_pdf_from_image_path = 'output-from-image.pdf'  # Path for image-to-PDF

# Call Node.js script for each action
execute_node_script('html-to-pdf', html_file_path, output_pdf_path)
execute_node_script('html-to-image', html_file_path, output_image_path)
execute_node_script('html-to-imgpdf', html_file_path, output_pdf_from_image_path)


# sendEmail.js dependency
nodemailer
googleapis
______________________________________________________________________________________________________________________________________________________________________________________________________________________________________