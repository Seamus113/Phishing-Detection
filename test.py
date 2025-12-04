from predict import predict

test_url = "http://bilibili.com"
lbl, p, feat = predict(test_url)
print("URL:", test_url)
print("Label:", lbl)
print("Prob phishing:", p)
print("Features:", feat)