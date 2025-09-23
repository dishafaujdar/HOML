from src.main import SpamClassifier
from src.performance_metrics import accuracy_score,precision_score,recall_score,f1_score

def test_spam_classifier():
    # Sample training data
    train_emails = [
        "free money now click here to win",
        "hello how are you doing today",
        "get rich quick scheme buy now",
        "meeting scheduled for tomorrow at 3pm",
        "congratulations you won million dollars",
        "thanks for your help yesterday",
        "urgent act now limited time offer",
        "can we reschedule our lunch meeting"
    ]
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

    # Sample test data
    test_emails = [
        "win money now",
        "let's meet for lunch",
        "urgent offer just for you",
        "see you tomorrow"
    ]
    expected = [1, 0, 1, 0]

    classifier = SpamClassifier(lr=0.1, epochs=500)
    predictions, probabilities = classifier.fit_and_predict(train_emails, train_labels, test_emails)

    acc = accuracy_score(expected, predictions)
    prec = precision_score(expected, predictions)
    rec = recall_score(expected, predictions)
    f1 = f1_score(expected, predictions)

    with open("test_output.txt", "w") as f:
        f.write(f"Predictions: {predictions}\n")
        f.write(f"Expected:    {expected}\n")
        f.write(f"Probabilities: {probabilities}\n")
        f.write(f"accuracy_score: {acc}\n")
        f.write(f"precision_score: {prec}\n")
        f.write(f"recall_score: {rec}\n")
        f.write(f"f1_score: {f1}\n")

    assert acc > 0.5, "Test failed: Accuracy below threshold."

if __name__ == "__main__":
    test_spam_classifier()