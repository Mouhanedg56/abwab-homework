# Outlier Detection Service

This service runs outlier detection given a text and a title. 

Only "Gift_Cards", "Digital_Music", "Magazine_Subscriptions", "Subscription_Boxes" categories are supported.

# Training

The training process depends on a rather chaotic pile of notebooks (due to time constrain) that retrieve and prepare data.
On each notebook, I tried different techniques and different features.

The technique used in production is the one used in **outlier-detection-text-only.ipynb** notebook.
We encoded text used embeddings and computed centroids. Then a radius was defined to detect outliers.

# Distribution drift
I used a very simple metric called z-score. To compute this metric, we need to subtract the mean and divide by the standard deviation for each feature.
The final score is the mean across all features. A high absolute value (typically > 2 or < -2) indicates that the point is an outlier for that particular feature.

# How to run the service
You can run the service directly without using docker
```bash
- poetry install
- poetry run uvicorn server:app --port 8000 --reload
```
To test the service go to browser and enter `http://localhost:8000/docs`. Click on **Try it out** and enter an example payload like this:
```json
{
  "auth_key": "secureapikey",
  "title": "A review",
  "text": "Good!",
  "category": "Gift_Cards"
}
```
To use Docker, run these commands:
```bash
- docker build . -t outlier_detector 
- docker run -d -p 5000:5000 outlier_detector
```
Then run this command to send a request to the API:
- Detect endpoint
```bash
curl -X POST http://localhost:5000/json/detect \
-H "Content-Type: application/json" \
-d '{
  "auth_key": "secureapikey",
  "title": "A review",
  "text": "Good!",
  "category": "Gift_Cards"
}'
```
- Distribution shift endpoint
```bash
curl -X POST http://localhost:5000/json/shift \
-H "Content-Type: application/json" \
-d '{
  "auth_key": "secureapikey",
  "title": "A review",
  "text": "Good!",
  "category": "Gift_Cards"
}'
```







