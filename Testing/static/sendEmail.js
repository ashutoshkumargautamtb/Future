// Load the nodemailer module
const nodemailer = require('nodemailer');
const { google } = require('googleapis');
const fs = require('fs');
const path = require('path');

// // SMTP configuration
// const smtpConfig = {
//   host: 'smtp.migadu.com', // Replace with your SMTP host (e.g., smtp.gmail.com)
//   port: 465, // Replace with your SMTP port (587 is commonly used for STARTTLS)
//   secure: true, // Set to true if you're using port 465
//   auth: {
//     user: 'admin@cryptonetworksupport.us', // Your email address
//     pass: 'luCifer@7838', // Your email password or app-specific password
//   },
// };

// // Create a transporter
// const transporter = nodemailer.createTransport(smtpConfig);

// // Email details
// const mailOptions = {
//   from: '"Your Name" <admin@cryptonetworksupport.us>', // Sender address
//   to: 'lavonnajefferson3@gmail.com', // List of recipients
//   subject: 'Hello from Node.js', // Subject line
//   text: 'This is a plain text email.', // Plain text body
//   html: '<b>This is an HTML email.</b>', // HTML body
// };

// // Send the email
// transporter.sendMail(mailOptions, (error, info) => {
//   if (error) {
//     return console.log('Error occurred:', error);
//   }
//   console.log('Email sent successfully:', info.response);
// });


// Path to the credentials.json and token.json files
const CREDENTIALS_PATH = path.join(__dirname, 'lavonnajefferson3@gmail.com.json');
const TOKEN_PATH = path.join(__dirname, 'token_lavonnajefferson3@gmail.com.json');

// Scopes for Gmail API
const SCOPES = ['https://www.googleapis.com/auth/gmail.send'];

// Load credentials and token
const credentials = JSON.parse(fs.readFileSync(CREDENTIALS_PATH));
const token = JSON.parse(fs.readFileSync(TOKEN_PATH));
const { client_secret, client_id, redirect_uris } = credentials.installed;

// Create an OAuth2 client and set the token
const oAuth2Client = new google.auth.OAuth2(
  client_id,
  client_secret,
  redirect_uris[0]
);
oAuth2Client.setCredentials(token);

// Send email using Gmail API
function sendEmail(auth) {
  const gmail = google.gmail({ version: 'v1', auth });

  const email = [
    'From: "Your Name" <lavonnajefferson3@gmail.com>',
    'To: lucifer.morningstarxx07@gmail.com',
    'Subject: Test Email from Gmail API',
    '',
    'This is a test email sent using the Gmail API!',
  ].join('\n');

  const encodedMessage = Buffer.from(email)
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');

  gmail.users.messages.send(
    {
      userId: 'me',
      requestBody: {
        raw: encodedMessage,
      },
    },
    (err, res) => {
      if (err) {
        console.error('Error sending email:', err);
        return;
      }
      console.log('Email sent successfully:', res.data);
    }
  );
}

// Start sending the email
sendEmail(oAuth2Client);