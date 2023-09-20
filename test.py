import pandas as pd
# from wand.image import Image as WandImg
from PIL import Image
import pytesseract
import pickle
from wordcloud import WordCloud
from stws import *


# text = """Writing and publishing have taken popularity on the internet using online services where text classification plays an important role (Luo 2021). An example where text classification can be applied is in the increasing amount of published research documents online or offline due to the advancement of computer and information technologies (Kim & Gil, 2019). Documents, in this case, refer to textual documents, and each document contains a group of words that ranges from sentence to paragraph long.  Using text classification, prediction and classification of documents can be made possible by categorizing them into which class they belong based on their inherent properties (Sarkar 2019).
# While there are many ways to classify research papers online, there is also a need to classify those with only physical copies. Approved research papers refer to peer-reviewed and panel-evaluated complete research in local school libraries. Additionally, research papers still in the proposal period are subject to revisions and are not easily distinguishable whether they are suitable for the course or not. For such reasons, it is necessary to make a tool that efficiently classifies approved and work-in-progress research papers.
# In this study, to classify research documents, first, OCR will be used. Optical Character Recognition (OCR) acquires an image through the use of a device, usually a camera or scanner, and then converts it to digital text(Holanda et al., 2018)(Goodrum et al., 2020). Then, supervised term weighting schemes are applied to assign a weight for each term in every document, enhancing text classification performance. (Alsmadi and Hoon 2018). Lastly, the documents will be assigned to their designated classes using different machine learning algorithms. (Raza et al., 2019)
# The proponents came up with the idea to design a classification model using a combination of supervised TWS and Machine Learning Algorithms, as well as prove which combination of Supervised TWS and Machine Learning Algorithm is the fastest for classifying our dataset. Finally, the proponents will develop a mobile application that will make use of the designed classifier model to categorize research papers and extract topics while utilizing the device’s camera for image acquisition.
# """
#
# lda = stws.LDA(text, 1, 10)
# topics = lda.getTopics()
# topics = " ".join(topics)
# print(topics)

#
# data = pickle.load(open(r"pickles/ABSTRACT SQRT TF EMONO MultinomialNB.pkl", "rb"))
# print(data)
#
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# data = pd.read_csv("Merged_Schools_Data.csv")
# category = data["Category"]
# print(category.drop_duplicates())

# fuck = data.drop(index=436)
#
# fuck.to_csv("Test.csv", index=False)

# results = pd.read_csv("Scores.csv")
# models = results["Model"]
# f1score = results["F1 Score"]
# def separateScores(dataset):
#     df = pd.DataFrame({"Model": models, "F1 Score": f1score}, index=results["Dataset"])
#     scores = df.drop(dataset)
#     return scores
#
# separateScores("ABSTRACT").to_csv("BG Scores.csv")
# req_image = []
# final_text = []
# path = r"d:\Users\users1\Documents\AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
# image_pdf = WandImg(filename=path, resolution=300)
# image_jpeg = image_pdf.convert('jpeg')
# for img in image_jpeg.sequence:
#     img_page = WandImg(image=img)
#     req_image.append(img_page.make_blob('jpeg'))
#
# for img in req_image:
#     txt = pytesseract.image_to_string(Image.open(img))
#     final_text.append(txt)
#
# print(final_text)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
filepath = r"d:\Users\users1\Pictures\research papers\fuck\ACLC 2\Capstone\Capstone_83.jpg"
ocr = pytesseract.image_to_string(Image.open(filepath))
print("OCR:", ocr)
tokens = getTokens(ocr)
print("STOPWORDS, FILTER BY 3, STEM:", tokens)
print("-------------------------------------------------------------------------------------------------------------------------------------------------")
topics = LDA(ocr, 1, 10).getTopics()
print("TOPICS:", topics)
