"""Chennai locality and merchant reference data used to personalize synthetic transactions."""

# (locality, lat, lon) -- approximate centroids
CHENNAI_LOCALITIES = [
    ("T Nagar", 13.0418, 80.2341),
    ("Anna Nagar", 13.0850, 80.2101),
    ("Velachery", 12.9756, 80.2207),
    ("Adyar", 13.0012, 80.2565),
    ("Mylapore", 13.0339, 80.2619),
    ("Thoraipakkam (OMR)", 12.9420, 80.2340),
    ("Guindy", 13.0067, 80.2206),
    ("Nungambakkam", 13.0569, 80.2425),
    ("Porur", 13.0382, 80.1565),
    ("Tambaram", 12.9249, 80.1000),
    ("Chrompet", 12.9516, 80.1462),
    ("Besant Nagar", 13.0002, 80.2666),
    ("Perambur", 13.1143, 80.2329),
    ("Ambattur", 13.1143, 80.1548),
    ("Pallavaram", 12.9675, 80.1491),
    ("Vadapalani", 13.0503, 80.2126),
    ("Kodambakkam", 13.0502, 80.2246),
    ("Royapettah", 13.0531, 80.2647),
    ("Thiruvanmiyur", 12.9830, 80.2591),
    ("Washermanpet", 13.1167, 80.2833),
]

# Other Indian cities used to simulate geo-impossible-travel fraud patterns.
OTHER_CITIES = [
    ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.7041, 77.1025),
    ("Bengaluru", 12.9716, 77.5946),
    ("Hyderabad", 17.3850, 78.4867),
    ("Kolkata", 22.5726, 88.3639),
    ("Singapore", 1.3521, 103.8198),
    ("Dubai", 25.2048, 55.2708),
]

# (merchant_name, category)
CHENNAI_MERCHANTS = [
    ("Nalli Silks - T Nagar", "clothing"),
    ("Pothys", "clothing"),
    ("Saravana Stores", "shopping_mall"),
    ("Chennai Citi Centre", "shopping_mall"),
    ("Phoenix MarketCity Chennai", "shopping_mall"),
    ("Express Avenue Mall", "shopping_mall"),
    ("Grand Sweets & Snacks", "food_dining"),
    ("Murugan Idli Shop", "food_dining"),
    ("Hotel Saravana Bhavan", "food_dining"),
    ("A2B - Adyar Ananda Bhavan", "food_dining"),
    ("Sangeetha Veg Restaurant", "food_dining"),
    ("Swiggy", "food_dining"),
    ("Zomato", "food_dining"),
    ("Chennai Metro Rail Recharge", "utility_bills"),
    ("TNEB Bill Payment", "utility_bills"),
    ("BWSSB Water Bill", "utility_bills"),
    ("Big Basket", "grocery"),
    ("More Supermarket", "grocery"),
    ("Reliance Fresh", "grocery"),
    ("Spencer's Retail", "grocery"),
    ("Nilgiris Super Market", "grocery"),
    ("Amazon India", "online_retail"),
    ("Flipkart", "online_retail"),
    ("Croma - Electronics", "electronics"),
    ("Reliance Digital", "electronics"),
    ("Vivek & Co", "electronics"),
    ("Apollo Pharmacy", "pharmacy"),
    ("MedPlus Pharmacy", "pharmacy"),
    ("IOC Petrol Bunk - Anna Salai", "fuel"),
    ("BPCL Fuel Station - OMR", "fuel"),
    ("Shell Petrol Bunk - Velachery", "fuel"),
    ("PVR Cinemas - Ampa Skywalk", "entertainment"),
    ("Sathyam Cinemas", "entertainment"),
    ("Chennai Super Kings Store", "entertainment"),
    ("IRCTC - Train Booking", "travel"),
    ("Ola Cabs", "travel"),
    ("Uber", "travel"),
    ("Chennai International Airport - Duty Free", "travel"),
    ("GRT Jewellers", "jewellery"),
    ("Joyalukkas", "jewellery"),
    ("Kalyan Jewellers", "jewellery"),
    ("Vijay Stationery & Books", "education"),
    ("Byju's Learning", "education"),
]

TAMIL_FIRST_NAMES_M = [
    "Arun", "Karthik", "Vignesh", "Suresh", "Senthil", "Prabhu", "Dinesh",
    "Bharath", "Aravind", "Mohan", "Rajesh", "Vijay", "Sathish", "Gokul",
    "Naveen", "Ramesh", "Saravanan", "Muthu", "Ganesh", "Hari",
]
TAMIL_FIRST_NAMES_F = [
    "Priya", "Kavitha", "Divya", "Lakshmi", "Meena", "Anitha", "Swathi",
    "Deepa", "Revathi", "Nithya", "Shalini", "Pooja", "Keerthana", "Sangeetha",
    "Vidya", "Gayathri", "Harini", "Preethi", "Janani", "Abirami",
]
TAMIL_LAST_NAMES = [
    "Subramaniam", "Krishnan", "Iyer", "Pillai", "Nair", "Raman", "Murthy",
    "Chandran", "Natarajan", "Venkatesan", "Rajan", "Gopal", "Sundaram",
    "Balasubramaniam", "Shankar", "Varadarajan",
]
