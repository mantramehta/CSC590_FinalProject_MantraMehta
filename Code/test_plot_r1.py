import matplotlib.pyplot as plt

datasets = ["4-way Additive", "2-way Epistatic"]
auc = [0.9804, 0.4784]
acc = [0.94, 0.46]

plt.figure(figsize=(6,4))
plt.bar(datasets, auc, color=["green", "red"])
plt.ylabel("ROC-AUC")
plt.title("Model Performance on Different Dataset Types")
plt.ylim(0,1)
plt.savefig("progress1_auc.png", dpi=300)
plt.show()
