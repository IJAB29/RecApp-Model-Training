import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

CSV_DIR = r"ALL_DATA(oversampled).csv"

docs = pd.read_csv(CSV_DIR, index_col=False)
background = np.array(docs["Backgrounds"])
category = np.array(docs["Categories"])
abstract = np.array(docs["Abstracts"])

new = ["""Writing and publishing have taken popularity on the internet using online services where text classification plays an important role (Luo 2021). An example where text classification can be applied is in the increasing amount of published research documents online or offline due to the advancement of computer and information technologies (Kim & Gil, 2019). Documents, in this case, refer to textual documents, and each document contains a group of words that ranges from sentence to paragraph long.  Using text classification, prediction and classification of documents can be made possible by categorizing them into which class they belong based on their inherent properties (Sarkar 2019).
While there are many ways to classify research papers online, there is also a need to classify those with only physical copies. Approved research papers refer to peer-reviewed and panel-evaluated complete research in local school libraries. Additionally, research papers still in the proposal period are subject to revisions and are not easily distinguishable whether they are suitable for the course or not. For such reasons, it is necessary to make a tool that efficiently classifies approved and work-in-progress research papers.
In this study, to classify research documents, first, OCR will be used. Optical Character Recognition (OCR) acquires an image through the use of a device, usually a camera or scanner, and then converts it to digital text(Holanda et al., 2018)(Goodrum et al., 2020). Then, supervised term weighting schemes are applied to assign a weight for each term in every document, enhancing text classification performance. (Alsmadi and Hoon 2018). Lastly, the documents will be assigned to their designated classes using different machine learning algorithms. (Raza et al., 2019)
The proponents came up with the idea to design a classification model using a combination of supervised TWS and Machine Learning Algorithms, as well as prove which combination of Supervised TWS and Machine Learning Algorithm is the fastest for classifying our dataset. Finally, the proponents will develop a mobile application that will make use of the designed classifier model to categorize research papers and extract topics while utilizing the device’s camera for image acquisition.
""", """HDS Bullding, 999 J.C. Aquino Avenue, Butuan City

ACLC College of Butuan City

Case Overview

On the first year, May 201 3-April 2014, the owner established standard
procedures and protocols regarding its physical environment and customer
service.

The KTV rooms of Lifestar Family Karaoke have noise insulation to
reduce passage of sound from one room to another. The KTV rooms have
two lighting options to choose from, the warm light bulb lighting or the disco
ball lighting with three settings. Before the customers enter the KTV room, the
employees must turn on the air conditioner and spray the room with air

freshener.

Aside from the KTV rooms, they also have a receiving area where the
customer can order, pay, and inquire for vacant rooms. The receiving area
has a sofa for customers to sit comfortably while waiting and is air
conditioned. The employees also play pop music playlists on the receiving
area. Regarding the lighting, they use white fluorescent light bulbs and mini
LED lights in violet and blue colors. A display case of expensive and imported

liquors can also be seen.

Lifestar also has a lobby; the lobby has tables and chairs for dining and
drinking, a mini stage for live band performances, and a big window which
allows light to come in; by the window, is the smoking area where there are
two chairs, a table, and an ashtray. The mini-stage has two microphones and
a sound system. The mini-stage is decorated with Gina cloth. The lobby has
no air conditioning. The lobby has blue LED light bulbs in each corner and

R
SCUFI AE Ae ceirurc\)e wm RifcmMres ANMIAICTMATING 2011
OC rl

RDO- , .
DO-ACAD-CS16-012 RESEARCH AND DEVELOPMENT OFFICE
HDS Building, 999 J.C. Aquino Avenue, Butuan City

d ACLC College of Butuan City

5
disco laser lights which emit blue, green, and red lights while the hallway has

dim lighting of blue, red, and green.

The employees must also clean the external area (street parking) and
internal (KTV rooms, lobby, receiving area, hallway, and kitchen) before and
after opening. Regarding the speed of service, when the customers are
finished with their orders, the employee must prepare the drinks immediately
and inform the kitchen staff (if they ordered food). They must also update the
customer the time that they must wait. They must also know the product,
price, and promotion of Lifestar to be able to respond to inquiries as well as
suggest. They must also know how to operate all of the equipment (KTV
machine, digital recording studio, television, sound system, microphone,
Xbox, and air conditioner). The technician also updates the song lists every
three months in the KTV room. In regards to responding to customer
complaints, the employees must do these right away: listen to the customer's
complaint, apologize, and provide a solution. Regarding the customer service
standard procedures and protocols, the employees must greet the customer
when they walk in and refer them as sir/ maâ€™am. For example: â€œWelcome to
Lifestar maâ€™am.â€ Employees must also pay attention what the customer wants
and be polite and respectful. Regarding the employeesâ€™ uniform, they are

required to wear pants, closed shoes, and a shirt provided by Lifestar.

Lifestar has a service bell in each KTV rooms. The service bell is used

by the customers to inform the employees that they need help. When the

RACUE!I Ne or Celene In RIIGINESS Annie rmatinn 2011
rr

RDO-ACAD-C516-017 RESFARCH AND DEVFLOPMENT OFFICE
HDS Building, 999 J.C. Aquino Avenue, Butuan City

d ACLC College of Butuan City

6
customer rings the bell, it automatically sends a notification of the room's

name to the receiving area; an employee will then go to that room.

The customers leave their comments, suggestions, and complaints in
the suggestion box and Lifestar's Facebook page. On the first year, according
to the suggestion box and Facebook page messages, the customers
appreciated that the employees are always wearing their uniform and closed
shoes and that they also greet them lively. They also said that the area and
KTV rooms were clean. The customers complained that sometimes the
employees forget to spray the air freshener after the previous customers

finished using the rooms. The sales during that year were Php 4,591,280.

On the second year, May 2014-April 2015, they renovated the lobby
area but they still operated during the renovation. After the renovation, what
used to be a mini stage was turned into a mini bar, an extra space for dining
was also made above the stairs, and the open window was turned into a
stage. The mini bar displays the different drinks that Lifestar offer, it also has
two stools and a counter top. The stage, used for live band performances and
open mics; has noise insulation, two microphones, two chairs, and a sound
system. A microphone is placed on the stage during an Open Mic, when
customers want to sing karaoke songs. Across the stage, is a LED flat screen
television, where singers and customers can see the lyrics of the song during
the Open Mic. Employees play music videos on the television when there are
no live bands and open mics. The lobby still has table and chairs for dining

and drinking. The lobby is air conditioned and the smoking area is outside the

RAcurinan ar SCIeNee ts MRIICMNECeS AMAMAICTM ATI Ne 201?
â€”â€”â€”â€”â€” IER E I SSR ee Re reee eee eee

RDO-ACAD-CS16-012 RESEARCH AND DEVELOPMENT OFFICE
ACLC College of Butuan City
@ HDS Building, 999 J.C. Aquino Avenue, Butuan City

7
entrance door of Lifestar. The smoking area has a sofa and an ashtray.
Regarding the lighting, they use disco lights and laser lights which emit green,
red, and blue colors. When there are only a few customers in the area, the air

conditioner is turned off and only the wall fan is turned on.

The protocol in cleaning the KTV was changed; employees must clean
the KTV room before and after the customers enter the room and not just
before the opening of the KTV bar. The employee turns on the air conditioner
of the room and then sprays each corner with air freshener, two to five
minutes before the customers enter the room. Employees are still required to
wear pants and closed shoes but instead of shirt provided by Lifestar, they

color code their t-shirts. Example, on Monday they wear pink.

That year, the service bell stopped working which causes the customer
to go to the receiving area when they need help. On the second year,
according to the suggestion box and Facebook messages, the customer
noticed a change that they started to clean the rooms before and after
customers used it and they appreciated it. They also commented on Lifestarâ€™s
renovation that they liked the new stage and the open mic but they still
preferred the brightness of the light on the previous year, they also
commented on atmosphere and said it was hot in the lobby because they will
turn off the air condition if there were only a few customers. The employees
were not wearing their color-coded uniform and they did not greet the
Customers. The sales during that year were: Php 3,021,659. The sales

decreased compared from the previous year.

RACHEI AP AF ericmee im mrcmeres annemicrnatian 207]

RDO-ACAD-CS16-012 RESEARCH AND DEVELOPMENT OFFICE
d ACLC College of Butuan City

HDS Building, 999 J.C. Aquino Avenue, Butuan City

8
On the third year, May 2015 - April 2016, nothing has changed with
Lifestarâ€™s standard procedures and a protocol regarding its customer service

and physical environment and the service bell is still not working.

On the third year, according to the suggestion box and Facebook
messages, the customer found their equipment has good performance; they
liked the sound quality of the microphones as well as the sound system. They
also felt that the television in the lobby is just the right size and has a nice
resolution that makes it easy for them to see the lyrics from the stage. They
liked that the song lists were always updated. They liked that it contains
popular songs of different genres. They noticed that Lifestarâ€™s employees
rarely greet them when they get in or do not even ask for repeat business.
The lack of friendly greeting from the employees gave them the impression
that the employees are not approachable. They said that not greeted by the
employees can put them in a bad mood and they will feel unimportant and
ignored. The customers also observed that the employees were not wearing
the uniform. They also commented on the strong scents on the rooms
because employees spray the room just right after they come in. They also
felt that instead of getting rid of the odor of the room, the smell of the air
freshener just mixes with the odor. They also complained about the hot
temperature in the lobby area. The customer observed that the employees
were not reliable with the time they promise to deliver the products; they also
noticed that they have slow service during peak hours. For example, instead

of five (5) minutes promise time, they will wait fifteen to twenty (15-20)

RASCvWrIiAP ne ceicuesâ€™ is RIICINE SS ANMINICTRATION 201?
rr

RDO-ACAD-CS16-012 RESEARCH AND DEVELOPMENT OFFICE
HDS Building, 999 J.C. Aquino Avenue, Butuan City

, ACLC College of Butuan City

9
minutes. They also felt scared or lonely because of the dark color and dim

lights. The sales decreased to Php 2,197,951.

The sales from 2013-2016 will be shown in the figure below:

Lifestar Family Karaoke
Comparative Annual Sales
_ 2013 - 2016

oe __ 8 4,591,280
____- 3,021,659
= 2,197,951

4,591,280 3,021,659 2,197,951

May 2013 â€” April 2014 May 2014 â€” April 2015 May 2015 â€” April 2016

Figure 2. Comparative Annual Sales of Lifestar Family Karaoke

Pacuriap ar cricncre In ricIiMvmecc ANLAINICTRATIAN 201)

a ee ee ee ee) dt eee Bo ee ed Oe he ae on YO ed ed ed

"""]


# tfmonobg = pd.read_csv("BACKGROUND TF MONO.csv", index_col=False).reset_index(drop=True, inplace=True)
# cv = CountVectorizer(stop_words="english", analyzer="word", vocabulary=tfmonobg)
def getConfidence(text, clf, vec):
    percentages = clf.predict_proba(vec.transform([text]))[0]
    confidence = np.sort(percentages)[-2] * 100
    confidence = format(confidence, '.2f')
    return f"{confidence}%"


path_stws = r"pickles/BACKGROUND SQRT TF IGM LinearSVC.pkl"
path_cv = r"pickles/Dictionary BACKGROUND SQRT TF IGM.pkl"

cv = pickle.load(open(path_cv, "rb"))
stws = pickle.load(open(path_stws, "rb"))

# X_train, X_test, y_train, y_test = train_test_split(cv.vocabulary, category, test_size=0.30, random_state=42)
# stws = SVC().fit(X_train, y_train)

for text in new:
    # conf = getConfidence(text, stws, cv)
    pred = stws.predict(cv.transform([text]))
    conf = np.sort(stws._predict_proba_lr(cv.transform([text]))[0])
    conf = [num * 100 for num in conf]
    print(pred)
    print(conf)
