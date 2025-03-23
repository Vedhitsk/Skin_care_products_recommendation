import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Import the Dataset 
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Skin Care Recommender System", page_icon=":rose:", layout="wide",)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
                icons=["house", "stars", "book"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Skin Care", "Get Recommendation", "Skin Care 101"],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Skin Care":
    st.title(f"{selected} Product Recommender :sparkles:")
    st.write('---') 

    st.write(
        """
        ##### **The Skin Care Product Recommendation Application is an implementation of Machine Learning that can provide recommendations for skin care products according to your skin type and problems.**
        """)
    
    #displaying a local video file

    video_file = open("skincare.mp4", "rb").read()
    st.video(video_file, start_time = 1) #displaying the video 
    
    st.write(' ') 
    st.write(' ')
    st.write(
        """
        ##### You will get recommendations for skin care products from various cosmetic brands with a total of 1200+ products tailored to your skin needs.
        ##### There are 5 categories of skin care products with 5 different skin types, as well as the problems and benefits you want to get from the product. This recommendation application is just a system that provides recommendations according to the data you enter, not a scientific consultation.
        ##### Please select the *Get Recommendation* page to start getting recommendations. Or select the Skin Care 101 page to see tips and tricks about skin care.
        """)
    
    st.write(
        """
        **Thank You :) !**
        """)
    
    
    st.info('Credit: Created by Somya, Shivang and Vedhit')

if selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    
    st.write(
        """
        ##### **To get recommendations, please enter your skin type, complaints, and desired benefits to get the right skin care product recommendations.**
        """) 
    
    st.write('---') 

    first,last = st.columns(2)

    # Choose a product product type category
    # pt = product type
    category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique() )
    category_pt = skincare[skincare['product_type'] == category]

    # Choose a skin type
    # st = skin type
    skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'] )
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    # pilih keluhan
    prob = st.multiselect(label='Skin Problems : ', options= ['Kulit Kusam', 'Jerawat', 'Bekas Jerawat','Pori-pori Besar', 'Flek Hitam', 'Garis Halus dan Kerutan', 'Komedo', 'Warna Kulit Tidak Merata', 'Kemerahan', 'Kulit Kendur'] )

    # Choose notable_effects
    # dari produk yg sudah di filter berdasarkan product type dan skin type(category_st_pt), kita akan ambil nilai yang unik di kolom notable_effects
    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    # notable_effects-notable_effects yang unik maka dimasukkan ke dalam variabel opsi_ne dan digunakan untuk value dalam multiselect yg dibungkus variabel selected_options di bawah ini
    selected_options = st.multiselect('Notable Effects : ',opsi_ne)
    # hasil dari selected_options kita masukan ke dalam var category_ne_st_pt
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Choose product
    # produk2 yang sudah di filter dan ada di var filtered_df kemudian kita saring dan ambil yang unik2 saja berdasarkan product_name dan di masukkan ke var opsi_pn
    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    # buat sebuah selectbox yang berisi pilihan produk yg sudah di filter di atas
    product = st.selectbox(label='Produk yang Direkomendasikan Untuk Kamu', options = sorted(opsi_pn))
    # variabel product di atas akan menampung sebuah produk yang akan memunculkan rekomendasi produk lain

    ## MODELLING with Content Based Filtering
    # Inisialisasi TfidfVectorizer
    tf = TfidfVectorizer()

    # Melakukan perhitungan idf pada data 'notable_effects'
    tf.fit(skincare['notable_effects']) 

    # Mapping array dari fitur index integer ke fitur nama
    tf.get_feature_names_out()

    # Melakukan fit lalu ditransformasikan ke bentuk matrix
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 

    # Melihat ukuran matrix tfidf
    shape = tfidf_matrix.shape

    # Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
    tfidf_matrix.todense()

    # Membuat dataframe untuk melihat tf-idf matrix
    # Kolom diisi dengan efek-efek yang diinginkan
    # Baris diisi dengan nama produk
    pd.DataFrame(
        tfidf_matrix.todense(), 
        columns=tf.get_feature_names_out(),
        index=skincare.product_name
    ).sample(shape[1], axis=1).sample(10, axis=0)

    # Menghitung cosine similarity pada matrix tf-idf
    cosine_sim = cosine_similarity(tfidf_matrix) 

    # Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama produk
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    # Melihat similarity matrix pada setiap nama produk
    cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

    # Membuat fungsi untuk mendapatkan rekomendasi
    def skincare_recommendations(nama_produk, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):

        # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
        # Dataframe diubah menjadi numpy
        # Range(start, stop, step)
        index = similarity_data.loc[:,nama_produk].to_numpy().argpartition(range(-1, -k, -1))

        # Mengambil data dengan similarity terbesar dari index yang ada
        closest = similarity_data.columns[index[-1:-(k+2):-1]]

        # Drop nama_produk agar nama produk yang dicari tidak muncul dalam daftar rekomendasi
        closest = closest.drop(nama_produk, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return df

    # Membuat button untuk menampilkan rekomendasi
    model_run = st.button('Find Other Product Recommendations!')
    # Mendapatkan rekomendasi
    if model_run:
        st.write('Here are recommendations for other similar products according to what you want')
        st.write(skincare_recommendations(product))
    
    
if selected == "Skin Care 101":
    st.title(f"Take a Look at {selected}")
    st.write('---') 

    st.write(
        """
        ##### **Here are some tips and tricks you can follow to maximize the use of skin care products.**
        """) 
    
    image = Image.open('imagepic.jpg')
    st.image(image, caption='Skin Care 101')
    

    
    st.write(
        """
        ### **1. Facial Wash**
        """)
    st.write(
        """
        **- Use facial wash products that have been recommended or that are suitable for you.**
        """)
    st.write(
        """
        **- Wash your face a maximum of 2 times a day, in the morning and at night before going to bed. Washing your face too often will remove the skin's natural oils. For those of you with dry faces, it's okay to just use plain water in the morning.**
        """)
    st.write(
        """
        **- Do not rub your face roughly as this can remove the skin's natural protective barrier.**
        """)
    st.write(
        """
        **- The best way to cleanse the skin is to use your fingertips for between 30-60 seconds in a circular and massaging motion.**
        """)
    
    st.write(
        """
        ### **2. Toner**
        """)
    st.write(
        """
        **- Use a toner that has been recommended or that is suitable for you.**
        """)
    st.write(
        """
        **- Pour the toner onto a cotton pad and gently wipe it onto your face. For maximum results, use 2 layers of toner, the first using a cotton pad and the last using your hands for better absorption.**
        """)
    st.write(
        """
        **- Use toner after washing your face**
        """)
    st.write(
        """
        **- For those of you with sensitive skin, as much as possible avoid skin care products that contain fragrance.**
        """)
    
    st.write(
        """
        ### **3. Serum**
        """)
    st.write(
        """
        **- Use a serum that has been recommended or that is suitable for you for maximum results.**
        """)
    st.write(
        """
        **- Serum is used after the face is completely clean so that the serum content is perfectly absorbed.**
        """)
    st.write(
        """
        **- Use the serum in the morning and at night before bed.**
        """)
    st.write(
        """
        **- Choose a serum according to your needs, such as removing acne scars or removing dark spots or anti-aging or other benefits.**
        """)
    st.write(
        """
        **- CHow to use serum so that it absorbs more perfectly is to pour it into the palm of your hand then pat it gently onto your face and wait until it is absorbed.**
        """)
    
    st.write(
        """
        ### **4. Moisturizer**
        """)
    st.write(
        """
        **- Use a moisturizer that has been recommended or that is suitable for you for maximum results.**
        """)
    st.write(
        """
        **- Moisturizer is a must-have skincare product because it is able to lock in moisture and various nutrients from the serum that has been used.**
        """)
    st.write(
        """
        **- For maximum results, use different moisturizers in the morning and at night. Morning moisturizers are usually equipped with sunscreen and vitamins to protect the skin from the harmful effects of UV rays and pollution, while night moisturizers contain various active ingredients that help the skin regeneration process while sleeping.**
        """)
    st.write(
        """
        **- Leave a time delay between using the serum and moisturizer for around 2-3 minutes to ensure the serum has been absorbed into the skin.**
        """)
    
    st.write(
        """
        ### **5. Sunscreen**
        """)
    st.write(
        """
        **- Use sunscreen that has been recommended or that is suitable for you for maximum results.**
        """)
    st.write(
        """
        **- Sunscreen is the main key to all skin care products because it protects the skin from the harmful effects of UVA and UVB rays, even blue light. All skin care products will not be felt if there is nothing to protect the skin.**
        """)
    st.write(
        """
        **- Apply sunscreen approximately the length of your index and middle fingers to maximize protection.**
        """)
    st.write(
        """
        **- Re-apply sunscreen every 2-3 hours or as needed.**
        """)
    st.write(
        """
        **- Keep using sunscreen even when you are indoors because sunlight from 10 o'clock onwards still penetrates through the windows and during cloudy weather.**
        """)
    
    st.write(
        """
        ### **6. Don't change skin care**
        """)
    st.write(
        """
        **Frequently changing skin care will cause facial skin to experience stress because it has to adapt to the product content. As a result, the benefits obtained are not 100%. Instead, use skin care products for months to see the results.**
        """)
    
    st.write(
        """
        ### **7. Consistent**
        """)
    st.write(
        """
        **KuThe key to facial care is consistency. Be diligent and persistent in using skin care products because the results are not instant.**
        """)
    st.write(
        """
        ### **8. Face is an asset**
        """)
    st.write(
        """
        **The various forms of humans are gifts given by the Creator. Take good care of that gift and be sincere as a form of gratitude. Choose products and care methods that suit your skin's needs. Using skin care products from an early age is the same as investing in old age.**
        """)
     
    
    
