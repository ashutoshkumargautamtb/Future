#Handle all api calls 
#Database Managment system(DBMS)

import pymysql

import pymysql  # For MySQL

# def insert_email_data(
#     email, content_txt, content_html, pdf_id, email_header, classification, psw, sending_method
# ):
#     """
#     Inserts a record into the email_data table.
    
#     Args:
#         email (str): The email address.
#         content_txt (str): The plain text content.
#         content_html (str): The HTML content.
#         pdf_id (str): The PDF URL or ID.
#         email_header (str): The email header information.
#         classification (str): The classification label.
#         psw (str): The PSW value.
#         sending_method (str): The method of sending (e.g., Email, SMS).
#     """
#     # Database connection details
#     host = "mydb.cpy2ykk2cppa.us-east-1.rds.amazonaws.com"
#     user = "admin"
#     password = "W7PeSBRI2s8AkAa"
#     database = "DB1"  # Replace with your schema name

#     try:
#         # Connect to the database
#         connection = pymysql.connect(host=host, user=user, password=password, database=database)
#         cursor = connection.cursor()

#         # SQL query to insert data
#         sql = """
#         INSERT INTO email_data (
#             Email, Content_TXT, Content_HTML, PDF_ID, Email_Header, Classification, PSW, sendingMethod
#         )
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
#         """

#         # Execute the query with provided arguments
#         cursor.execute(sql, (email, content_txt, content_html, pdf_id, email_header, classification, psw, sending_method))
#         connection.commit()  # Commit the transaction

#         print("Record inserted successfully!")

#     except Exception as e:
#         print(f"Error: {e}")

#     finally:
#         if connection:
#             connection.close()  # Close the database connection


# # Example Data
# example_email = "example@domain.com"
# example_content_txt = "Hi there! This is a plain text example."
# example_content_html = "<p>Hi there! This is an HTML example.</p>"
# example_pdf_id = "https://aws.amazon.com/example-pdf-12345.pdf"
# example_email_header = "Sample Email Header"
# example_classification = "INB"
# example_psw = "NA"
# example_sending_method = "Email"

# # Insert Example Record
# insert_email_data(
#     email=example_email,
#     content_txt=example_content_txt,
#     content_html=example_content_html,
#     pdf_id=example_pdf_id,
#     email_header=example_email_header,
#     classification=example_classification,
#     psw=example_psw,
#     sending_method=example_sending_method
# )

def delete_email_data(record_id):
    """
    Deletes a record from the email_data table based on its ID.
    
    Args:
        record_id (int): The ID of the record to delete.
    """
    # Database connection details
    host = "mydb.cpy2ykk2cppa.us-east-1.rds.amazonaws.com"
    user = "admin"
    password = "W7PeSBRI2s8AkAa"
    database = "DB1"  # Replace with your schema name

    try:
        # Connect to the database
        connection = pymysql.connect(host=host, user=user, password=password, database=database)
        cursor = connection.cursor()

        # SQL query to delete the record
        sql = "DELETE FROM email_data WHERE ID = %s;"

        # Execute the query
        cursor.execute(sql, (record_id,))
        connection.commit()  # Commit the transaction

        print(f"Record with ID {record_id} deleted successfully!")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if connection:
            connection.close()  # Close the database connection


# Example: Deleting a Record by ID
record_id_to_delete = 1  # Replace with the actual ID of the record you want to delete
delete_email_data(record_id_to_delete)