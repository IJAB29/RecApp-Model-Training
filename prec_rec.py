import pandas

from salitaulap import *

models = [
"TF MONO KNeighborsClassifier",
"TF MONO MultinomialNB",
"TF MONO LinearSVC",
"SQRT TF MONO KNeighborsClassifier",
"SQRT TF MONO MultinomialNB",
"SQRT TF MONO LinearSVC",
"TF EMONO KNeighborsClassifier",
"TF EMONO MultinomialNB",
"TF EMONO LinearSVC",
"SQRT TF EMONO KNeighborsClassifier",
"SQRT TF EMONO MultinomialNB",
"SQRT TF EMONO LinearSVC",
"TF IGM KNeighborsClassifier",
"TF IGM MultinomialNB",
"TF IGM LinearSVC",
"SQRT TF IGM KNeighborsClassifier",
"SQRT TF IGM MultinomialNB",
"SQRT TF IGM LinearSVC"
]
bg_prec = [
    0.781659458,
    0.564104215,
    0.866617705,
    0.872462368,
    0.911549223,
    0.943297765,
    0.814942808,
    0.563746108,
    0.866617705,
    0.877361535,
    0.908969015,
    0.93861922,
    0.75210601,
    0.898966255,
    0.927540028,
    0.83028373,
    0.925652885,
    0.933909418,
]
ab_prec = [
    0.797330445,
    0.576546545,
    0.903361503,
    0.877848725,
    0.910247093,
    0.943446485,
    0.790057153,
    0.57610578,
    0.895578013,
    0.87571233,
    0.909650305,
    0.940984033,
    0.853674195,
    0.913255005,
    0.931482283,
    0.891095985,
    0.918299608,
    0.934908228,
]

bg_rec = [
    0.758196003,
    0.597599638,
    0.79807299,
    0.850178758,
    0.853057378,
    0.936248265,
    0.786288233,
    0.594882245,
    0.79807299,
    0.85796204,
    0.850339988,
    0.933530875,
    0.732702198,
    0.844146022,
    0.912930775,
    0.828550765,
    0.908327698,
    0.930674692,
]
ab_rec = [
    0.752194575,
    0.635416668,
    0.860701865,
    0.844245163,
    0.86169793,
    0.93636149,
    0.765350548,
    0.62749094,
    0.852721355,
    0.80887681,
    0.86607821,
    0.933644097,
    0.820944888,
    0.881104738,
    0.927009243,
    0.889645633,
    0.893410745,
    0.933304425,
]

ab_df = pandas.DataFrame({"Model": models, "Precision": ab_prec, "Recall": ab_rec})
bg_df = pandas.DataFrame({"Model": models, "Precision": bg_prec, "Recall": bg_rec})

ab_df.plot(x="Model", kind="bar", ylim=(.55, 1))
plt.show()
bg_df.plot(x="Model", kind="bar", ylim=(.55, 1))
plt.show()

# visualize({"Model": models, "Precision": ab_prec}, "Model", "Precision")
# visualize({"Model": models, "Precision": bg_prec}, "Model", "Precision")
#
# visualize({"Model": models, "Recall": ab_rec}, "Model", "Recall")
# visualize({"Model": models, "Recall": bg_rec}, "Model", "Recall")
