const fs = require('fs');
const path = require('path');
const pdf = require('html-pdf-node');
const puppeteer = require('puppeteer');
const { PDFDocument } = require('pdf-lib');

// Function to generate PDF from HTML
async function generatePdfFromHtml(htmlSource, outputPdfPath) {
    try {
        const options = { format: 'A4' };
        const file = htmlSource.startsWith('<') ? { content: htmlSource } : { url: `file://${path.resolve(htmlSource)}` };

        const pdfBuffer = await pdf.generatePdf(file, options);
        fs.writeFileSync(outputPdfPath, pdfBuffer);
        console.log(`PDF successfully created: ${outputPdfPath}`);
    } catch (err) {
        console.error('Error creating PDF:', err);
    }
}

// Function to generate an image from HTML
async function generateImageFromHtml(htmlSource, outputImagePath) {
    try {
        // Launch Puppeteer
        const browser = await puppeteer.launch();
        const page = await browser.newPage();

        if (htmlSource.startsWith('<')) {
            await page.setContent(htmlSource, { waitUntil: 'networkidle0' });
        } else {
            const normalizedPath = `file://${path.resolve(htmlSource)}`;
            await page.goto(normalizedPath, { waitUntil: 'networkidle0' });
        }

        // Take a screenshot and save it
        await page.screenshot({ path: outputImagePath, fullPage: true });
        console.log(`Image successfully created: ${outputImagePath}`);

        // Close the browser
        await browser.close();
    } catch (err) {
        console.error('Error creating image:', err);
    }
}

// Function to convert an image to a PDF
async function convertImageToPdf(imagePath, outputPdfPath) {
    try {
        // Read the image file
        const imageBytes = fs.readFileSync(imagePath);

        // Create a new PDF document
        const pdfDoc = await PDFDocument.create();

        // Embed the image into the PDF
        const image = await pdfDoc.embedPng(imageBytes); // Use embedJpg if the image is in JPG format
        const page = pdfDoc.addPage();

        // Set the page size to match the image dimensions
        page.setSize(image.width, image.height);

        // Draw the image onto the page
        page.drawImage(image, {
            x: 0,
            y: 0,
            width: image.width,
            height: image.height,
        });

        // Serialize the PDF to bytes
        const pdfBytes = await pdfDoc.save();

        // Write the PDF to a file
        fs.writeFileSync(outputPdfPath, pdfBytes);
        console.log(`PDF created from image: ${outputPdfPath}`);
    } catch (err) {
        console.error('Error converting image to PDF:', err);
    }
}

// Main function
(async () => {
    const args = process.argv.slice(2);
    const action = args[0]; // Action: html-to-pdf, html-to-image, or html-to-imgpdf
    const htmlSource = args[1]; // HTML file path or source code
    const outputPath = args[2]; // Output file path

    if (!action || !htmlSource || !outputPath) {
        console.error('Usage: node script.js <action> <htmlSource> <outputPath>');
        console.error('Actions: html-to-pdf, html-to-image, html-to-imgpdf');
        process.exit(1);
    }

    try {
        if (action === 'html-to-pdf') {
            await generatePdfFromHtml(htmlSource, outputPath);
        } else if (action === 'html-to-image') {
            await generateImageFromHtml(htmlSource, outputPath);
        } else if (action === 'html-to-imgpdf') {
            const tempImagePath = path.join(path.dirname(outputPath), 'temp-output.png');
            await generateImageFromHtml(htmlSource, tempImagePath);
            await convertImageToPdf(tempImagePath, outputPath);
            fs.unlinkSync(tempImagePath); // Remove temporary image file
        } else {
            console.error('Invalid action. Use html-to-pdf, html-to-image, or html-to-imgpdf.');
        }
    } catch (err) {
        console.error('Error:', err);
    }
})();
