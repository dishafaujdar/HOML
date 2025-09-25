from src.BinaryClassification import SpamClassifier
from src.performance_metrics import accuracy_score,precision_score,recall_score,f1_score

def test_spam_classifier():
    # More complex training data
    train_emails = [
        "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now.",
        "Dear friend, I hope you are doing well. Let's catch up soon.",
        "Limited time offer! Buy one get one free on all products. Hurry up!",
        "Your Amazon order has been shipped. Track your package here.",
        "You have been selected for a free cruise to the Bahamas. Call now!",
        "Meeting rescheduled to next Monday at 10am. Please confirm your availability.",
        "Earn money from home. No experience required. Start today!",
        "Lunch at the new Italian place tomorrow?",
        "Get cheap medicines online without prescription. Order now!",
        "Project deadline extended to next Friday. Update your tasks accordingly.",
        "Winner! You have been chosen for a special prize. Respond immediately.",
        "Can you send me the notes from yesterday's class?",
        "Lowest prices on electronics. Visit our website for exclusive deals.",
        "Let's go hiking this weekend if the weather is good.",
        "Urgent: Your account has been compromised. Reset your password now.",
        "Family dinner at my place this Sunday. Let me know if you can make it.",
    ]
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

    # More complex test data
    test_emails = [
        "Special offer just for you! Buy two get one free. Limited time only.",
        "Let's meet at the coffee shop at 4pm.",
        "Your PayPal account has been suspended. Click here to verify.",
        "Don't forget to bring your laptop for the meeting.",
        "Congratulations! You are the lucky winner of our lottery.",
        "Can you review my code before tomorrow's deadline?",
        "Cheap flights to Europe available now. Book your tickets today!",
        "Looking forward to our trip next month.",
    ]
    expected = [1, 0, 1, 0, 1, 0, 1, 0]

    classifier = SpamClassifier(lr=0.1, epochs=1000)
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