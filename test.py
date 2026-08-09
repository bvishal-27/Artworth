import joblib
import pandas as pd

# ==============================
# 1. Load Trained Model
# ==============================
print("Loading saved model...")
model = joblib.load("best_model.pkl")
print("✅ Model loaded successfully!")

# ==============================
# 2. Define Feature Order (same as training)
# ==============================
feature_columns = [
    "Artist Reputation", "Height", "Width", "Weight",
    "Creation_Year", "Artwork_Age", "Previous_Auction_Price",
    "Sculpture_Type", "Material", "Artist Name",
    "Artist_Alive", "Aesthetic_Descriptor", "Provenance"
]

# ==============================
# 3. Create Two Test Inputs
# ==============================
# Input 1: Famous artist, large bronze sculpture, high auction history
input_1 = {
    "Artist Reputation": 0.95,        # very high reputation
    "Height": 200,                    # large sculpture
    "Width": 80,
    "Weight": 300,
    "Creation_Year": 1880,
    "Artwork_Age": 143,               # old sculpture
    "Previous_Auction_Price": 500000, # strong past auction price
    "Sculpture_Type": "Classical",
    "Material": "Bronze",
    "Artist Name": "Auguste Rodin",
    "Artist_Alive": "No",
    "Aesthetic_Descriptor": "Elegant",
    "Provenance": "Auctioned in Paris"
}

# Input 2: Lesser-known artist, small wood carving, no auction history
input_2 = {
    "Artist Reputation": 0.20,        # low reputation
    "Height": 50,
    "Width": 20,
    "Weight": 15,
    "Creation_Year": 2015,
    "Artwork_Age": 8,                 # recent artwork
    "Previous_Auction_Price": 0,      # no auction record
    "Sculpture_Type": "Contemporary",
    "Material": "Wood",
    "Artist Name": "Unknown Artist",
    "Artist_Alive": "Yes",
    "Aesthetic_Descriptor": "Simple",
    "Provenance": "Private Collection"
}

# ==============================
# 4. Convert to DataFrame
# ==============================
test_df = pd.DataFrame([input_1, input_2], columns=feature_columns)

# ==============================
# 5. Make Predictions
# ==============================
predictions = model.predict(test_df)

print("\n=== Prediction Results ===")
for i, pred in enumerate(predictions, start=1):
    print(f"Input {i} predicted value (relative index): {pred:.2f}")

# ==============================
# Explanation:
# - Input 1 should yield a much higher predicted value due to
#   high artist reputation, old age, bronze material, and past auctions.
# - Input 2 should yield a much lower predicted value because of
#   unknown artist, recent creation, wood material, and no auction history.
# ==============================

