# Description
Description: "Quintessentia" aims to make it easier for users to understand and navigate podcasts by providing summaries and timestamps. The project involves taking audio podcasts and converting them to text, then punctuating and chunking the text into meaningful parts. It also involves summarizing each part and timestamping it to allow users to easily navigate the podcast. The project creates a chart of the best topics in the podcast and provides a summary of the entire podcast.

# Main functionality of the API
The upload function allows you to upload an audio file and get its transcript in text format. It takes an optional parameter file of type UploadFile which is a file that has been uploaded to the server.

The function first gets the audio file name and checks if it already exists in the current directory. If it doesn't, the function saves the file locally and then calls the google_transcribe function to get the transcription of the audio file. It then calls the get_colored_transcript function to highlight certain parts of the transcript, such as names, products, companies, and dates. The function then saves the transcript locally and returns it to the user.

If the audio file already exists in the current directory, the function reads the ready transcript from the local file and returns it to the user.

The chunking_text function takes a block of text and splits it into reasonable chunks. It takes a parameter body of type Body, which is a dictionary containing the text to be chunked. The function first extracts the text from the input and then calls the create_embedding function to split the text into sentences and get their embeddings. It then calls the create_df function to create a dataframe with the sentences and the generated timestamps. Finally, it calls the get_middle_points function to get the points where the text should be split and then chunks the text accordingly. It returns the chunked text as a list of paragraphs.
