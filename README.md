# Business Card Reader App

The main idea of this project is that extracting entities from the scanned Business Card.

![cover](https://user-images.githubusercontent.com/30235603/199545111-2801a9f2-f3f8-48fc-a6f7-71b0dd91df32.png)

### Project Features:
- Extract Entities (text and data) from image of Business Card
    - Entities : Name, Organization, Phone, Email and Web Address

### Tasks:
- 1-Location of Entity
- 2-Text of Corresponding Entity

### Examples:
- Name:
- Designation
- Organization
- Phone
- Email
- Web Address

![1](https://user-images.githubusercontent.com/30235603/199517908-a8b6a70a-96da-4fb2-8f65-7d23248e99f7.png)


### Technologies:

- Computer Vision
    - Scanning Document
	- Identify Location of Text
	- Extract the Text from Image

Using OpenCV and Tesseract OCR

----
- Natural Language Processing
	- Extract Entities from Text
	- Cleaning and Parsing

Using Pandas, spaCy, RegEx

---
### Stages of Development

**1-Setting up Project**
- Installations

**2-Data Preparation**
- Extract Text and Location from Business Card

**3-Labelling**
- BIO Tagging

**4-Data Preprocessing**
* Text Cleaning and Processing

**5-Training Named Entity Recognition**
* Train Machine Learning Model

**6-Prediction**
* Parsing and Bounding Box

**7-Document Scanner App**
* Automatic Document Scanner App

---

### Architecture

**Business Card** -> **Extract Text from Image Using OCR** -> **Text** -> **Text Cleaning** -> **Deep Learning Model Trained in spaCy for NER** -> **Entities**

### Training Architecture

**Collected Data** -> **Extract Text from Image Using OCR** -> **Text** -> **Labeling** -> **Text Cleaning** -> **Train NER Model in SpaCy**

---
# Installations

### Environment Installation
```sh
conda create -n docscanner python=3.9
```
```sh
activate docscanner
```
```sh
pip install -r requirements.txt
```
If you do not use anaconda, **type this:**
```sh
python -m venv docscanner 
```
**Activation:**
```sh
.\docscanner\Scripts\activate
```
**For Linux or Mac:**
```sh
source <venv>/bin/activate
```
```sh
pip install -r requirements.txt
```
---

### Install Tesseract OCR and Pytesseract

####  Installation for Tesseract OCR 
https://tesseract-ocr.github.io/tessdoc/Installation.html

**For windows:**
https://digi.bib.uni-mannheim.de/tesseract/

**And download this:**
tesseract-ocr-w32-setup-v4.1.0.20190314.exe

**Note that:**
When you install Tesseract OCR, save the url where it is installed. It will be required in environmental setup.

![2](https://user-images.githubusercontent.com/30235603/199517914-a38b3392-472a-4827-a30d-5641fe8f11ac.png)

After installation of tesseract, check “Environment Variables”

Click Path, and check the url. If the urls are not there, you can manually add them into environment variables.

![3](https://user-images.githubusercontent.com/30235603/199517919-eecdae20-9809-426b-b6a9-c761ad94cbb2.png)

#### Installation of PyTesseract

After this installation, go terminal and type 
```sh
pip install pytesseract
```

#### Instalation of spaCy

Go this website, 
https://spacy.io/usage

For Windows
```sh
pip install -U spacy
python -m spacy download en_core_web_sm
```

---
---
# Section 1 - Data Preparation with PyTesseract
## Notebook: 01_PyTesseract.ipynb
Open a page from Jupyter Notebook and import all libraries that we installed
![4](https://user-images.githubusercontent.com/30235603/199517924-81cd50e5-df66-42d7-b505-35b850a2d779.png)
and it works without any errors!

### Hierarchy of PyTesseract - How it works -

There are 5 levels in PyTesseract.

* Level 1 
This is for defining the page. If there is only one image, then it is only one level.

* Level 2
It defines the block.

* Level 3
It defines the paragraph.

* Level 4
This is for line.

* Level 5
It is for words.

> First, in Level 1 it will **define the page.**
> In that page, it will **define the block**
> and then in that block, it will **detect paragraph.** 
> In paragraph, it will **detect all line**
> and in line, it will **detect words.**
> Then it will **detect letters from words.**

After all these steps, it will take each letters to Machine Learning model.

#### **Level 1 - Page**
In this case, we only have one image which means, there is only one page.
![5](https://user-images.githubusercontent.com/30235603/199517926-d2b6b4ee-4bcd-4681-bea8-b26b38b67eb0.png)

#### **Level 2 - Block**
![6](https://user-images.githubusercontent.com/30235603/199517930-1bf38e56-27e7-430f-915b-7941747485c4.png)

#### **Level 3 - Paragraph**
![7](https://user-images.githubusercontent.com/30235603/199517935-e969f4bc-3cc9-481e-aca7-6072a2b96fe9.png)

#### **Level 4 - Line**
![8](https://user-images.githubusercontent.com/30235603/199517939-54a96d00-382e-4c8d-9aa7-0338f337c676.png)

#### **Level 5 - Words**
![9](https://user-images.githubusercontent.com/30235603/199517945-dfb0c115-326c-4f20-a21d-5b4dfcc52653.png)


After all these steps, it will detect all letters ( I am kinda lazy for framing each words here :) but I will draw them, you will be able to find them below)

After letters are detected, machine learning model will classify it what kind of alphabet or number etc.

## Section 1.2

Now, we will get the hierarchy from image to data using PyTesseract.

In order to get those information, there is a special command called **image_to_data** in PyTesseract.

When you execute the command, here is what happens:
![10](https://user-images.githubusercontent.com/30235603/199517949-b62df25d-9d05-4924-94fb-25ac66f14a0b.png)

And now, I will split the data for each line
![11](https://user-images.githubusercontent.com/30235603/199517993-fc185d65-e9d6-482d-8907-62f1c69697ee.png)
![12](https://user-images.githubusercontent.com/30235603/199517995-66240752-9bbd-4b40-8ce3-eb68f2894783.png)


Now what I will do is that I will take every element from there that I listed and I will also split by backslash “\t” and will create a new list.
![13](https://user-images.githubusercontent.com/30235603/199518761-1ff46249-b99a-4d7e-b302-ed3dd2085c26.png)
Here it is seen, first element is separated, I will apply this for all elements.
![14](https://user-images.githubusercontent.com/30235603/199518767-dfe7065f-97fa-4dc3-83a9-a4c23d3d542b.png)
![15](https://user-images.githubusercontent.com/30235603/199518772-47290659-5d24-420d-b54a-288c1c10ee81.png)
And I will turn them into a Data Frame
![16](https://user-images.githubusercontent.com/30235603/199518780-79fdf281-8eb7-474a-85cd-8ba985fab3bd.png)

You should also notice that one of the columns is called **Level**, it is what I mentioned before. 
Also **Level** defines the **block numbers.**
And this is how we extract data from image to pandas Data Frame. Through this, we have much clear information.

In order to show, I will draw boxes according to the positions by considering what **Level** means.
- **Level 2: Block**
- **Level 3: Paragraph**
- **Level 4: Line number**
- **Level 5: Text**

![17](https://user-images.githubusercontent.com/30235603/199518782-d0a5862b-e88c-4865-838d-3bfd417bb235.png)

Before drawing boxes, I should handle these missing values and types to proper form.

 **1- Drop Missing Values**
**2- Turn the Columns into integer**

![18](https://user-images.githubusercontent.com/30235603/199518785-81f4cfdf-c7aa-433a-b4f0-ee92bd296c05.png)

#### Drawing:

- **l: level**
- **x: left**
- **y: top**
- **w: width**
- **h: height**
- **c: confidence score**

![19](https://user-images.githubusercontent.com/30235603/199518790-4978e814-5f55-4ff5-b898-ee5df7027d9d.png)


![20](https://user-images.githubusercontent.com/30235603/199518793-14cfb964-fa50-4499-8465-729fbf118583.png)
This is what PyTesseract does :)

---
---

# Section 2 - Data Preprocessing and Preparation
## Notebook: 02_Data_Preparation.ipynb

Now what I am going to do is, I will apply all these steps to all dataset.

For this, I will open a new notebook and will import libraries that I will use. By using **glob**, I will get paths of images and by using **os** I will separate filename 
![21](https://user-images.githubusercontent.com/30235603/199518794-d64da6b2-0970-491f-8829-a26c1cda8c79.png)

Like what I did in first notebook, I will also do same things and I will get a DataFrame
![22](https://user-images.githubusercontent.com/30235603/199518795-26ab22f9-951c-4d5d-8bb8-9571cf3cddd4.png)

But now, I will only get those which their conf. is grater than 30 and I will create a new DataFrame called **businessCard**.
![23](https://user-images.githubusercontent.com/30235603/199518799-7eda72ea-e258-4698-b05f-5d92ace45aab.png)
![24](https://user-images.githubusercontent.com/30235603/199518804-872457ef-39c7-4100-860d-7f8d147e63bf.png)

Here is the result.

I did this steps to see what will happen. Looks super! Now I will apply all these steps to all data.

![25](https://user-images.githubusercontent.com/30235603/199518805-0ecdd4d0-0ead-4256-857e-4de8ba605c08.png)

After getting a new DataFrame called **allBusinessCard** I will save it into csv file.

Next step is, I will label this data, **for example: name, organization, phone number etc.**

### Labeling

Now, what I will do is tagging all words in the cvs file.

### BIO / IOB Format

**SOURCE:** https://medium.com/analytics-vidhya/bio-tagged-text-to-original-text-99b05da6664
The BIO / IOB format (short for inside, outside, beginning) is a common tagging format for tagging tokens in a chunking task in computational linguistics (ex. named-entity recognition). The B- prefix before a tag indicates that the tag is the beginning of a chunk, and an I- prefix before a tag indicates that the tag is inside a chunk. The B- tag is used only when a tag is followed by a tag of the same type without O tokens between them. An O tag indicates that a token belongs to no entity / chunk. 

The following figure shows how a BIO tagged sentence looks like:

![26](https://user-images.githubusercontent.com/30235603/199518809-266ea236-0e06-49be-b338-e66c74dbf558.png)

### Entities

| Description|Tag |
|----------|:-------------:|
|Name	|NAME|
|Designation |DES|
|Organization|ORG|
|Phone Number|PHONE|
|Email Address|EMAIL|
|Website|WEB|

![27](https://user-images.githubusercontent.com/30235603/199518816-6e564739-3313-43bd-b309-cbb51476770e.png)

Unfortunately, there is no shortcuts of tagging. I have to do this manually inside of the csv file.

After this long and boring process, I will prepare the data for the training.

---
---

# Section 3 - Data Preprocessing and Cleaning
## Notebook: 03_Data_Preprocessing.ipynb

### 1-Data Preprocessing

### **SpaCy Data Format:**
![28](https://user-images.githubusercontent.com/30235603/199518821-479de5ec-1b5d-4d56-baaa-331ff8105230.png)

In this example from tha documentation of SpaCy , there are totally 11 arrows, as it is seen in the example which is **[(0, 11, “BUILDING”)]** It means, the buildings which is **“Tokyo Tower”** it starts from index **0** to **11**st.
That’s what I need to do for preparing the data, I will determine them like this.

> Link: https://spacy.io/usage/training#basics

Before starting, convert csv file to tsv (tab separated value). In order to convert the csv file, just click “Save As” and choose **Tab Delimited txt file.**

And, time to open the file

![29](https://user-images.githubusercontent.com/30235603/199518823-4ba3d43a-96d2-4a7a-8095-cf4704bbb689.png)

When we look at the data, it will look like this,

![30](https://user-images.githubusercontent.com/30235603/199518827-5ad75bcb-4bdf-49d4-b227-ff9be2c69d6e.png)

I will apply same methods what I did in 2nd notebook

![31](https://user-images.githubusercontent.com/30235603/199518830-e066bb17-5af4-4c62-aa8c-88086642968f.png)

This is the data what I have right now. I will also turn this into a pd DataFrame

![32](https://user-images.githubusercontent.com/30235603/199518834-f5f4dbd5-814e-43d3-89d5-72fac1383383.png)

### **2-Data Cleaning**

This section will be cleaning process. In this case, I will remove white spaces and unwanted special characters because I don’t need them.

First, I will define white space, there are different ways to define that but the useful way is doing it with “string” library.

Next thing is defining special characters. But here I will not remove all special characters. For instance “@” that is important for Email.

![33](https://user-images.githubusercontent.com/30235603/199518836-164aabef-e0a6-47c7-b2ff-15e2a3db3ac2.png)

In the above image, I also defined a function which remove white spaces and special characters.

I will apply this function to the DataFrame

![34](https://user-images.githubusercontent.com/30235603/199518840-1c47d466-4b75-4097-9d37-c926cf146509.png)

Next thing what I will do is, convert the data into SpaCy format.

### **Converting to SpaCy Format:**

![35](https://user-images.githubusercontent.com/30235603/199518844-c53576cd-269a-4ab5-81b3-7616695a3087.png)
![36](https://user-images.githubusercontent.com/30235603/199518850-0236d627-f6c1-4814-8588-bcb210195f85.png)
![37](https://user-images.githubusercontent.com/30235603/199518852-5a58ae1e-870b-4470-85b5-23280fb01a5a.png)

Basically what I try to create is that content is all information in the text, annotations is about the labels and their start and end positions.

I am not into **“O”** because it means **outside.** I am only into **“B”** and **“I”**

![38](https://user-images.githubusercontent.com/30235603/199518855-28ad41e1-3072-42d3-95bb-dfbac53e8c17.png)

Lets check if annotation is correct.

![39](https://user-images.githubusercontent.com/30235603/199518861-d3104a06-f47f-4364-9f44-dd7ba8a3098b.png)


as it is seen, start and end positions of phone are correct.

After this step, now I will apply these steps to all dataset.

![40](https://user-images.githubusercontent.com/30235603/199518862-ebaba23a-1785-4d41-a086-45bd6cd3e1dc.png)
![41](https://user-images.githubusercontent.com/30235603/199518865-39d2c81f-dbeb-44eb-a086-b92a557859aa.png)
![42](https://user-images.githubusercontent.com/30235603/199518868-8b60a587-29f8-4ef5-864d-327b45b0e0f4.png)


### **Splitting Data**

First I will shuffle the dataset

![43](https://user-images.githubusercontent.com/30235603/199518870-a75c7e84-e165-4b71-9dbe-341aac19b9c4.png)

And then, I will split the data 90% - 10%

![44](https://user-images.githubusercontent.com/30235603/199518873-6ab64f13-1a8b-434c-9118-83a7db54f90a.png)


Next thing is saving data into data folder by using “pickle” library.

![45](https://user-images.githubusercontent.com/30235603/199518875-ba013dae-af51-4a8d-8c12-eac62d711180.png)

In the next step, I will train a Named Entity Recognition **(NER)** model.

---
---

# Section 4 - Train Named Entity Recognition (NER)
## Code: preprocess . py

Spacy is one of the most popular and useful framework for Natural Language Processing. It is easy to use and it is a way to find a lot of predefined models.
> https://spacy.io/
> https://spacy.io/usage/training

What I will do is, take the model, use the framework and training. It is very simple.

In order to get the model, first visit this website and get Quickstart. Choose what you need, then SpaCy will give you the predefined code.

![46](https://user-images.githubusercontent.com/30235603/199518881-ef5e08e0-b568-435c-9e15-7c4c9db5f2db.png)

Then click download. That’s all!

In order to fill all the details of configuration, I need to type magical word to terminal

When I open this config file, there is a note which is:

```sh
python -m spacy init fill-config ./base_config.cfg ./config.cfg
```

![47](https://user-images.githubusercontent.com/30235603/199518885-356c065c-771e-4a31-a2ee-e67d7d329a79.png)

I will paste it to terminal.

![48](https://user-images.githubusercontent.com/30235603/199518888-744b953a-d7eb-4381-b552-4d8196e9a27f.png)

![49](https://user-images.githubusercontent.com/30235603/199518664-16419110-db07-4417-a0e3-da638406425f.png)


And it worked!

Now I will train the model by following commands.

![50](https://user-images.githubusercontent.com/30235603/199518669-3e7d7207-4021-43fe-9e39-f200e92d9289.png)

As you see here, the format is **.spacy** but before I saved train and test data as **.pickle**. Now I will convert them into **.spacy**
For doing this, in the documentation there is a section called **Preparing Training Data** It is also so easy to convert. I will copy the code and will just make some changes. That’s all!

![51](https://user-images.githubusercontent.com/30235603/199518672-c901124a-338f-4685-b567-75ac92cd3363.png)

All I need to do is just run the preprocess.py file which will make converting process from **.pickle** to **.spacy** format.

![52](https://user-images.githubusercontent.com/30235603/199518675-4bef8c46-d294-4ad4-9ff0-76d213051123.png)

And It’s ready, that’s all here!

![53](https://user-images.githubusercontent.com/30235603/199518676-5850a3a7-51cf-45ea-975d-e77eb178ab9c.png)

And as a final step in the training process, I need to train the model using the config file.
Before this, I will create a folder called output to save output files.

```sh
python -m spacy train config.cfg --output output --paths.train data/train.spacy --paths.dev data/test.spacy
```

When I run this code, the training is start

![54](https://user-images.githubusercontent.com/30235603/199518686-fee38879-ae4f-43e7-b690-4a4175cccada.png)

After the training, there became two folder inside of the output file.

![55](https://user-images.githubusercontent.com/30235603/199518740-9ef5a1b5-1c95-43aa-a42a-b34ad0a6f451.png)

There are two folders which are those above the image. 
**model-best** file contains **high score model** which has 0.64 score, 
**model-last** contains **the last one** which has 0.62 score after the training.

I will use the best one for prediction. 

---
---

# Section 5- Prediction
## Notebook: 04_Predictions.ipynb

In this section, I will test the NER model that I trained using **SpaCy**. All steps that I will apply are the same with that before I did. 

From old notebook, I will copy and paste the function for cleaning text.

![56](https://user-images.githubusercontent.com/30235603/199518741-178ea69b-a5fc-482d-9312-b375e4b95ac8.png)


### STEPS:

#### 1- Load NER Model
![57](https://user-images.githubusercontent.com/30235603/199518745-ee99a274-ae33-46ac-82b0-e750d02c2d89.png)

#### 2- Load Image
![58](https://user-images.githubusercontent.com/30235603/199518750-c7ff2c6c-7a17-4a5f-a977-8ad7e0896c19.png)

#### 3- Extract Data from Text using Pytesseract
![59](https://user-images.githubusercontent.com/30235603/199518753-e49b7d34-ec6d-4ecd-a66b-5f5cb072d3e3.png)
#### 4- Convert into DataFrame
![60](https://user-images.githubusercontent.com/30235603/199518757-d795a93c-3240-48ae-a991-9ba59e5fadba.png)

#### 5- Convert Data into Content
![61](https://user-images.githubusercontent.com/30235603/199521113-445accb5-2b1b-430e-afd7-0c7a30edef64.png)

#### 6- Get Predictions from NER Model and Render the Content
There are some ways to render the content
![62](https://user-images.githubusercontent.com/30235603/199521114-d5330fd7-653a-41d4-b941-ae46a2807455.png)

#### 7- Render the Content
![63](https://user-images.githubusercontent.com/30235603/199521118-7169d7a2-940f-48f6-9890-82d2176d1ceb.png)

Here it is seen, the NER (name entity recognition) model classified the content.

Now I will tag every word again.

### Tagging
There are different ways to do **BIO Tagging** but I will use the same doc.
![64](https://user-images.githubusercontent.com/30235603/199521121-4a748d05-6484-45ec-ae24-4d9f84886f06.png)

Token means every words.
I will convert this information to DataFrame.
![65](https://user-images.githubusercontent.com/30235603/199521130-22d45a4b-2ad2-4bf1-b245-9b7bb662509f.png)

I also turned token into DataFrame and now I will combine it with **doc_text**
![66](https://user-images.githubusercontent.com/30235603/199521133-739329fa-6592-42c0-a837-47575b1dc8e9.png)

And this is basically what lambda function does:
![67](https://user-images.githubusercontent.com/30235603/199521136-338d5777-61f6-4987-8320-6ac60c30d936.png)

After this step, I will add one more column to the DataFrame which is entities.
![68](https://user-images.githubusercontent.com/30235603/199521140-6c5731c4-5025-4ec6-b90b-cb9e4fb1614b.png)
![69](https://user-images.githubusercontent.com/30235603/199521143-2076016e-5f52-4fce-b2ba-cde664745ddc.png)

Here you can see, there are some **“NaN”** values, I will replace them with **“O”**

![70](https://user-images.githubusercontent.com/30235603/199521148-2c720cce-a1cd-4ce7-847c-fa5676ee117f.png)

As next step, I will combine “label” with data_clean column.
![71](https://user-images.githubusercontent.com/30235603/199521152-7fca6029-7764-4a02-8359-20ece755affe.png)

If I make this joining, it will be much convenient for drawing bounding boxes.
![72](https://user-images.githubusercontent.com/30235603/199521156-041e3575-1182-4e3a-a644-15c3fab4a5a9.png)

The reason why adding “+1” is that words are separated by one space.
If I do cumulative sum, I can get the end position of every words and minus one is removing space.
Here is the correct end positions.
![73](https://user-images.githubusercontent.com/30235603/199521159-bd79198f-5c57-42ad-82d4-22c55f213e52.png)

I will also create start position, in order to get start position is end position - length of word.
![74](https://user-images.githubusercontent.com/30235603/199521161-e319d81e-0b20-42c2-99c3-bb81b8bc63e0.png)

Now I will combine them inside of **df_clean**
![75](https://user-images.githubusercontent.com/30235603/199521163-697ac583-0837-4351-9247-ffb06cc4daaf.png)

And now, I will merge all dataframes into a new dataframe
![76](https://user-images.githubusercontent.com/30235603/199521165-7196b6ad-617a-46b2-ac1f-43afd28724bd.png)

Text and token may look like same but let’s check again by looking at last columns of the DataFrame
![77](https://user-images.githubusercontent.com/30235603/199521167-b0bd6f6c-d624-49e7-a2f2-7163a991a903.png)

As it is seen, token contains clear text.

### Bounding Box

In order to draw bounding box what I have to do is that I need to take the information except the label O. 
For this, I will filter the main dataframe.
![78](https://user-images.githubusercontent.com/30235603/199521173-267b9e7a-e4e3-4cec-b7a0-1c5bcaf25459.png)

And result,
![79](https://user-images.githubusercontent.com/30235603/199521178-4e86ab15-2be9-4e3e-9ba2-e29ae569e9ef.png)

As next step, I will combine BIO information.

For this, I will separate labels by applying lambda function which removes first value of label.
![80](https://user-images.githubusercontent.com/30235603/199521186-9432c277-034a-4b63-a64e-568e56f923be.png)

And now I will define a class which groups texts if they are same.
![81](https://user-images.githubusercontent.com/30235603/199521191-07eb504e-59f0-41f3-b856-0f87f71fa7ac.png)
![82](https://user-images.githubusercontent.com/30235603/199521193-a43debd7-c49f-4c21-9d13-514bf3574578.png)


According to this info, I will draw boxes again. But before doing this I will create two more columns for right and bottom positions.
![83](https://user-images.githubusercontent.com/30235603/199521195-779ef5f5-41a0-47de-b463-5b376f502b9c.png)
![84](https://user-images.githubusercontent.com/30235603/199521199-f884045e-e335-46e1-acb3-879e587c0afd.png)

For tagging, I will also groupby the dataframe by group

![85](https://user-images.githubusercontent.com/30235603/199521203-3fae1794-0300-4268-b5e9-de2acc1d3eb0.png)

> **NOTE:** I changed the image in order to get more clear value. So, next data will be different than last ones.

![86](https://user-images.githubusercontent.com/30235603/199521207-2e529bf7-6810-461a-915c-8b5566cfc62f.png)

![87](https://user-images.githubusercontent.com/30235603/199542954-fdc60ce5-84b0-4d1f-a0fb-9cbcef41e915.png)

And entities are drawn.

Now I will combine the text where B - I tags are. At the same time I will also do parsing. 
For example for the phone number, I will only take digits, for e-mail address, I will only take special characters etc.

![88](https://user-images.githubusercontent.com/30235603/199543058-ff1d76e7-f229-4465-b7ca-f4c5b4564349.png)
![89](https://user-images.githubusercontent.com/30235603/199543065-337a0e02-4b18-4517-be4c-fafbf78c3330.png)

It works well! It is cleaning special characters and this is how parser will work.
Now by using the entities I will save them into a dictionary, for this I will open a basic loop. 

![90](https://user-images.githubusercontent.com/30235603/199543070-9a39d23c-96db-498a-ad08-120209bf0da1.png)

The basic idea is, for instance in the image above, B-NAME: james, I-NAME: bond, what I will do is combine them.

![91](https://user-images.githubusercontent.com/30235603/199543076-2421a8e1-7f90-4226-b9e9-d7e9bafe98c5.png)

The result:
![92](https://user-images.githubusercontent.com/30235603/199543079-a9faa08c-2a24-4806-b669-17a396a8a2cb.png)

Except phone number, everything looks great.

Now in order to proper all codes, I will define a pipe which has all these steps and prediction function.

From **notebook: 04_Predictions.ipynb** I copied all steps and created them as a function.
I only deleted some usefulness lines and codes. Nothing changed actually.

You can find the code in prediction.py

And notebook: **05_Final_Predictions.ipynb** it is my test notebook, I test my prediction function which is prediction.py

![93](https://user-images.githubusercontent.com/30235603/199543083-5562ff12-b73e-4069-abda-c20040279960.png)

And here are the results:
### **Test 1 of Version 1**
![94](https://user-images.githubusercontent.com/30235603/199543090-649e83ad-2f78-4fac-b3e6-9233dc2c288b.png)
![95](https://user-images.githubusercontent.com/30235603/199543110-a4f046b1-7514-4f71-963b-8a2cf290ef4a.png)


As you can realize, the model confused in detecting phone again :) 

### **Test 2 of Version 1**
![96](https://user-images.githubusercontent.com/30235603/199543114-4605feec-48d7-499e-8b1a-eec436afdb07.png)

### **Test 3 of Version 1**
![97](https://user-images.githubusercontent.com/30235603/199543508-bfc1393e-f664-4a07-8cd6-7bcd3211aa61.png)

### **NOTE:**
I have created a folder called **VERSION_2**. In this folder I only made change inside of **clean_text function**. In the first version, text are turning **lowercase**, but in the **Version 2** I have **canceled** this. **Because some organizations etc have uppercase words**, that is why I have canceled and it worked slightly better. I am not saying this, accuracy is saying.
Here is the accuracy reports:

![98](https://user-images.githubusercontent.com/30235603/199543520-0a9a84a9-2995-4a89-b5d2-e253952eb878.png)

> In the **first version the best accuracy was 0.64**, but **here, it is 0.72**. Much better!

Here are some examples of predictions.

### **Test 1 of Version 2**
![99](https://user-images.githubusercontent.com/30235603/199543525-c13ded69-0a3d-4321-ba81-25385488e293.png)

### **Test 2 of Version 2**
![100](https://user-images.githubusercontent.com/30235603/199543536-a15a013c-5548-4026-925e-c42174d65d51.png)

### **Test 3 of Version 2**
![101](https://user-images.githubusercontent.com/30235603/199543543-bca7778d-58c3-4f07-9821-3dd24df15e24.png)

---
---

# Section 6 - Document Scanner
## Notebook: Document_Scanner.ipynb

In this notebook, I will work on fixing images which are rotated etc. Because in order to work with PyTesseract in proper way, it is necessary. PyTesseract does not work well with rotated images.

![102](https://user-images.githubusercontent.com/30235603/199543549-7426ef16-0d5b-4555-8b2b-d710250f686b.png)


### Steps:
**1- Resize the image and set aspect ratio**
**2-Image Processing**
- **Enhance**
- **Gray Scale**
- **Blur**
- **Edge Detection**
- **Morphological Transform**
- **Contours**
- **Find Four Points**

### 1-Resize the image and set aspect ratio
![103](https://user-images.githubusercontent.com/30235603/199543587-a8d8d221-cfe1-4916-82a6-d1ae5ea45356.png)

### 2-Image Processing
#### Enhance
![104](https://user-images.githubusercontent.com/30235603/199543593-c4e204c8-cb24-45e5-995e-8ad08825a313.png)

#### Edge Detection
![105](https://user-images.githubusercontent.com/30235603/199543595-3205eb32-aa4e-46f6-aa97-46598bf895fc.png)

As you can realize, there are some noises around the image, I will apply morphological functions to clean them. 

After Dilation, here is the result, as it is seen, thickness is increase, as my 2.step I will apply closing
![106](https://user-images.githubusercontent.com/30235603/199543598-57af9b63-8114-4afc-9fb5-9c620e3dd85c.png)

#### Closing
![107](https://user-images.githubusercontent.com/30235603/199543603-77751a01-6806-4e8f-b711-7e11ad98bfc8.png)

Now I will find the contours.
![108](https://user-images.githubusercontent.com/30235603/199543605-ac2fe01a-f491-4b55-be8f-872057283b11.png)

What I will do is that I will multiply this four_points with the multiplier which is width of the original image divided by width of the resize image.
![109](https://user-images.githubusercontent.com/30235603/199543645-c30bb3b7-0a31-4f8b-ba25-20e83feec617.png)

After these four points, I will wrap the original image using imutils library.
![110](https://user-images.githubusercontent.com/30235603/199543649-1a2cb168-afcc-4d84-aafd-4b9c10168512.png)

#### And it is time to define a function which does all these steps
![111](https://user-images.githubusercontent.com/30235603/199543654-93d9f45d-cb2a-4d4b-90fb-7362dd150842.png)

In order to analyse the images I also return resized image (which is drawn its contours) and closing image.

##### Here is an example:
![112](https://user-images.githubusercontent.com/30235603/199543663-5c9ddbf5-d3fe-44fc-97dc-8cb606f70371.png)

##### Another Example:
![113](https://user-images.githubusercontent.com/30235603/199543667-5bb50376-d8a0-4f5c-978d-3b62b5672d27.png)

##### Another Example:
![114](https://user-images.githubusercontent.com/30235603/199543673-9e526f95-0a82-431f-9f56-037040d3022b.png)

##### Another Example:
![115](https://user-images.githubusercontent.com/30235603/199543679-1bf9cc39-3b07-4072-9537-09770913805e.png)

As next step, I will also define a function for **finding great brightness and contrast**.
![116](https://user-images.githubusercontent.com/30235603/199543683-2f3e59a4-63f7-4502-961e-49d343fbb4a9.png)

##### Here is its example:
![117](https://user-images.githubusercontent.com/30235603/199543686-3f00e66b-9f45-4655-b60e-4ddb8d6fcdc0.png)
As it can be seen, color balance of magic image is much clear, when I apply NER algorithm it will be easy to read and detect.

##### Another example:
![118](https://user-images.githubusercontent.com/30235603/199543689-1a99c041-3c5f-474f-a016-61023ce8a0d3.png)

As summary of Magic Image is that the function increases the contrast and brightness of image.

##Integration of NER Prediction

First thing what I do is that, I import my best model which is in the **version 2** and then, I read one of the images

#### Here are results:
![119](https://user-images.githubusercontent.com/30235603/199543696-b22298c6-9739-4a73-a729-5af3020d71ea.png)
![120](https://user-images.githubusercontent.com/30235603/199543699-932bce79-868e-4666-85ac-eb9300b25f2a.png)

![121](https://user-images.githubusercontent.com/30235603/199543703-0bd89d22-be27-492a-8a64-2d0dca810852.png)

![122](https://user-images.githubusercontent.com/30235603/199543710-0ef1291d-ef89-47ee-b90b-3bfa5099ea55.png)

![123](https://user-images.githubusercontent.com/30235603/199543716-e77b4887-ac05-4e5f-a7ae-fed5939027f2.png)


> Unfortunately, **the model can not predict well** because of that **I didn’t feed it with more data** and that’s the result. If I give more data, I would definitely get much better results. 
But this is not my priority, **I am doing this exercise to learn and practise with PyTesseract.**
