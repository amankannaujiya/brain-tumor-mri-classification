import os
import requests

search_terms = ['car', 'house', 'dog', 'text', 'cat', 'nature', 'city', 'people', 'random', 'flower']
save_dir = './BrainTumorData/unknown'
os.makedirs(save_dir, exist_ok=True)

count = 1

for term in search_terms:
    for i in range(10):  # 10 images per term
        url = f"https://source.unsplash.com/300x300/?{term}&sig={i}"
        try:
            img_data = requests.get(url).content
            with open(os.path.join(save_dir, f'{term}_{i}.jpg'), 'wb') as handler:
                handler.write(img_data)
            print(f"✅ Downloaded {term}_{i}.jpg")
            count += 1
        except Exception as e:
            print(f"❌ Failed {term}_{i}: {e}")

