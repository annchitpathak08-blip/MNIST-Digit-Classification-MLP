**Testing Different architectures for same model**



1. **Making a smaller model by lowering number of neurons in hidden1 and hidden 2**



**784 → 64 → 32 → 10**

Epoch \[1/5] - Loss: 0.4029 - Train Accuracy: 88.78%

Epoch \[2/5] - Loss: 0.1800 - Train Accuracy: 94.77%

Epoch \[3/5] - Loss: 0.1308 - Train Accuracy: 96.11%

Epoch \[4/5] - Loss: 0.1041 - Train Accuracy: 96.86%

Epoch \[5/5] - Loss: 0.0851 - Train Accuracy: 97.42%



Test Accuracy: 97.01%

Saved: training\_loss.png

Saved: confusion\_matrix.png

Saved: misclassified.png



Model saved as mnist\_mlp.pth



\[Inference Test] True label: 7 | Predicted label: 7

Saved: single\_prediction.png

END



2\. **Making a bigger model by increasing number of neurons in hidden1 and hidden 2**



**784 → 256 → 128 → 10**

Epoch \[1/5] - Loss: 0.2850 - Train Accuracy: 91.60%

Epoch \[2/5] - Loss: 0.1069 - Train Accuracy: 96.75%

Epoch \[3/5] - Loss: 0.0706 - Train Accuracy: 97.80%

Epoch \[4/5] - Loss: 0.0505 - Train Accuracy: 98.40%

Epoch \[5/5] - Loss: 0.0385 - Train Accuracy: 98.73%



Test Accuracy: 97.65%

Saved: training\_loss.png

Saved: confusion\_matrix.png

Saved: misclassified.png



Model saved as mnist\_mlp.pth



\[Inference Test] True label: 7 | Predicted label: 7

Saved: single\_prediction.png

END



3\. **Making a shorter model by removing hidden 1 entirely**



**784 → 128 → 10**

Epoch \[1/5] - Loss: 0.3477 - Train Accuracy: 90.69%

Epoch \[2/5] - Loss: 0.1603 - Train Accuracy: 95.40%

Epoch \[3/5] - Loss: 0.1083 - Train Accuracy: 96.86%

Epoch \[4/5] - Loss: 0.0808 - Train Accuracy: 97.67%

Epoch \[5/5] - Loss: 0.0640 - Train Accuracy: 98.15%



Test Accuracy: 97.52%

Saved: training\_loss.png

Saved: confusion\_matrix.png

Saved: misclassified.png



Model saved as mnist\_mlp.pth



\[Inference Test] True label: 7 | Predicted label: 7

Saved: single\_prediction.png

END











***runtime in all about 30 secs***

