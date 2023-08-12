images = soup.find_all('img')

for image in images[-1]:

name = image['alt']

link = image['src']

with open(name.replace(' ', '-').replace('/', '') + '.jpg', 'wb') as f:

im = requests.get(link)

f.write(im.content)


image_ids = train_df['image_id'].unique()
print(len(image_ids))
valid_ids = image_ids[-10:]
train_ids = image_ids[:-10]
# valid and train df
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]


