
# Sentiment and Aspect prediction
## Abstract
The problem has two subtasks: predicting aspect and polarity (or sentiment). For polarity task, the assumption that one text has only one polarity is made. However, for aspect prediction, one text can contain multiple aspects. The experiment starts with baseline model which is a logistic regression model with bag-of-word feature. Then, more sophisticated models are developed, including RNN, CNN, and Fine-tuned transformers, to find the best candidate for each task.
The candidates of each task are then combined to create a Sentiment-Aspect prediction system.

---
Project report [PDF], explaining experiment processes, model selection and discussion, is available in this [link](https://production-gradescope-uploads.s3-us-west-2.amazonaws.com/uploads/pdf_attachment/file/68615294/Contest-1-report.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIGIENPBVZV37ZJPA%2F20220416%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220416T072501Z&X-Amz-Expires=10800&X-Amz-SignedHeaders=host&X-Amz-Signature=5e41624c31a61c5609f1f0cf7ab7fa9e5f215548cf81f15d92c9c49df83e1005)