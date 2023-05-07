"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import streamlit.components.v1 as components  # html extensions
# st.set_page.config(layout='wide', initial_sidebar_state='expanded')
from streamlit_option_menu import option_menu
import base64
import shap
import shap
import lime

from streamlit_javascript import st_javascript

# Data handling dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest

# App declaration
def main():
    # st.sidebar.markdown('side')
    st.markdown(
        """
        <style>
        .reportview-container {
        background: url('resources/imgs/sample.jpg')
        }
        .sidebar .sidebar-content {
        background: url('resources/imgs/sample.jpg')
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    with st.sidebar:
        from PIL import Image
        image2 = Image.open('resources/imgs/fraud.jpg')
        st.image(image2, caption='Data Analytics')

        page_selection = option_menu(
            menu_title=None,
            options=["Overview", "Step 3: Output", "Step 2: Model", "Step 1: Input"],
            icons=['file-earmark-text', 'graph-up', 'robot', 'file-earmark-spreadsheet'],
            menu_icon='cast',
            default_index=0,
            # orientation='horizontal',
            styles={"container": {'padding': '0!important', 'background_color': 'red'},
                    'icon': {'color': 'red', 'font-size': '18px'},
                    'nav-link': {
                        'font-size': '15px',
                        'text-align': 'left',
                        'margin': '0px',
                        '--hover-color': '#4BAAFF',
                    },
                    'nav-link-selected': {'background-color': '#6187D2'},
                    }
        )
        st.info('This algorithm predicts which financial statements are anomalous and explains why. '
                'An explanation is given about the data that is used (input), '
                'the algorithm that is used (model) and about the predictions of the algorithm (output)', icon="ℹ️")
    # page_options = ["Recommender System","Movie Facts","Exploratory Data Analysis","About"]




    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    # page_selection = st.sidebar.radio("Choose Option", page_options)
    if page_selection == "Overview":
        # Header contents
        st.write('## Financial Fraud Detection Application')
        st.write('### Detecting and Preventing Fraud with Financial Statements')
        st.markdown("This app is designed to flag potentially fraudulent financial statements submitted by companies with an insurance guarantee. By using a statistical (or other) model to predict and flag fraudulent statements, the app provides credit underwriters with an extra variable to consider when assessing credit risk. "
                    "The app analyzes the predicted fraud across different buckets such as industry, year, and financial type, providing insights into the appropriateness of the data for modeling purposes. It also generates a fraud indicator (or probability) that can be incorporated into the credit risk model. "
                    )

        age = st.slider('**Percentage of anamolous transections to display?**', 0 , 100)


        col1, col2 = st.columns([3, 1])
        data = np.random.randn(10, 1)

        col1.subheader("Selected statements")
        col1.line_chart(data)

        col2.subheader("Summary statistics")
        col2.write(data)





    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Step 2: Model":
        # Header Contents
        st.write("# Anomaly Detection Models")

        if st.checkbox("Algorithm Explanation"):
            filters = ["Benford's Law","Isolation Forest", "Local Outlier Factor"]
            filter_selection = st.selectbox("**Algorithms**", filters)
            if filter_selection == "Benford's Law":
                st.write("## Benford's Law Analysis")
                st.image('resources/imgs/ben.png')
                st.markdown("After this step, the tree would look as follows:")


            if filter_selection == "Isolation Forest":
                st.write("## Isolation Forest Algorithm")
                st.markdown("Isolation Forest is an unsupervised machine learning algorithm for anomaly detection. "
                            "As the name implies, Isolation Forest is an ensemble method (similar to random forest). "
                            "In other words, it use the average of the predictions by several decision trees when assigning the final anomaly score to a given data point. "
                            "Unlike other anomaly detection algorithms, which first define what’s “normal” and then report anything else as anomalous, "
                            "Isolation Forest attempts to isolate anomalous data points from the get go.")
                st.subheader("The algorithm")
                st.markdown("Suppose we had the following data points:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image('resources/imgs/is1.png', use_column_width=True)
                    st.markdown(
                        "The isolation forest algorithm selects a random dimension (in this case, the dimension associated with the x axis) and randomly splits the data along that dimension.")
                with col2:
                    st.image('resources/imgs/is2.png', use_column_width=True)
                    st.markdown(
                        "The two resulting subspaces define their own sub tree. In this example, the cut happens to separate a lone point from the remainder of the dataset. The first level of the resulting binary tree consists of two nodes, one which will consist of the subtree of points to the left of the initial cut and the other representing the single point on the righ")
                with col3:
                    st.image('resources/imgs/is3.png', use_column_width=True)
                    st.markdown(
                        "It’s important to note, the other trees in the ensemble will select different starting splits. In the following example, the first split doesn’t isolate the outlier.")
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.image('resources/imgs/is4.png', use_column_width=True)
                    st.markdown(
                        "We end up with a tree consisting of two nodes, one that contains the points to the left of the line and the other representing the points on the right side of the line.")
                with col5:
                    st.image('resources/imgs/is5.png', use_column_width=True)
                    st.markdown(
                        "The process is repeated until every leaf of the tree represents a single data point from the dataset. In our example, the second iteration manages to isolate the outlier.")
                with col6:
                    st.image('resources/imgs/is6.png', use_column_width=True)
                    st.markdown("After this step, the tree would look as follows:")

                col7, col8, col9 = st.columns(3)
                with col7:
                    st.image('resources/imgs/is7.png', use_column_width=True)
                    st.markdown(
                        "Remember that a split can occur along the other dimension as is the case for this 3rd decision tree.")
                with col8:
                    st.image('resources/imgs/is8.png', use_column_width=True)
                    st.markdown(
                        "On average, an anomalous data point is going to be isolated in a bounding box at a smaller tree depth than other points.")
                with col9:
                    st.image('resources/imgs/is9.png', use_column_width=True)
                    st.markdown(
                        "When performing inference using a trained Isolation Forest model the final anomaly score is reported as the average across scores reported by each individual decision tree.")

            if filter_selection == "Local Outlier Factor":
                st.write("## Local Outlier Factor Algorithm")
                st.subheader("What is LOF?")
                st.markdown(
                    "Local outlier factor (LOF) is an algorithm that identifies the outliers present in the dataset. But what does the **local outlier** mean?")
                st.markdown(
                    "When a point is considered as an outlier based on its local neighborhood, it is a local outlier. LOF will identify an outlier considering the density of the neighborhood. LOF performs well when the density of the data is not the same throughout the dataset.")
                st.markdown("To understand LOF, it is important to have an understanding of the following concepts:")
                st.markdown("- Distance and K-neighbors")
                st.markdown("- Reachability distance (RD)")
                st.markdown("- Local reachability density (LRD)")
                st.markdown("- Local Outlier Factor (LOF)")
                st.subheader("Distance and K-neighbors")
                st.markdown(
                    "K-distance is the distance between the point, and it’s Kᵗʰ nearest neighbor. K-neighbors denoted by Nₖ(A) includes a set of points that lie in or on the circle of radius K-distance. K-neighbors can be more than or equal to the value of K. How’s this possible?")
                st.markdown("Below an example is given. Let’s say we have four points A, B, C, and D (shown below).")
                st.image('resources/imgs/lo1.png', width=500,
                         caption='K-distance of A with K=2')
                st.markdown(
                    "If K=2, K-neighbors of A will be C, B, and D. Here, the value of K=2 but the ||N₂(A)|| = 3. Therefore, ||Nₖ(point)|| will always be greater than or equal to K.")
                st.subheader("Reachability distance (RD)")
                st.latex(r'''
                            RD( X_{i}   , X_{J} )  =  max(K  -  distance(X_{i}), distance(X_{i}, X_{J}))     
                            ''')
                st.markdown(
                    "It is defined as the maximum of K-distance of Xj and the distance between Xi and Xj. The distance measure is problem-specific (Euclidean, Manhattan, etc.)")
                st.image('resources/imgs/lo2.png', width=500,
                         caption='Illustration of reachability distance with K=2')
                st.markdown(
                    "In layman terms, if a point Xi lies within the K-neighbors of Xj, the reachability distance will be K-distance of Xj (blue line), else reachability distance will be the distance between Xi and Xj (orange line).")
                st.subheader("Local reachability density (LRD)")
                st.latex(r'''
                           LRD_{k}(A) =  \frac{1}{ \sum x_{j} \epsilon N_{k} \frac{RD(A,X_{j})}{ \| N_{K}(A) \| }} 
                            ''')
                st.markdown(
                    "LRD is inverse of the average reachability distance of A from its neighbors. Intuitively according to LRD formula, more the average reachability distance (i.e., neighbors are far from the point), less density of points are present around a particular point. This tells how far a point is from the nearest cluster of points. :blue[Low values of LRD implies that the closest cluster is far from the point.]")

                st.subheader("Local Outlier Factor (LOF)")
                st.markdown(
                    "LRD of each point is used to compare with the average LRD of its K neighbors. LOF is the ratio of the average LRD of the K neighbors of A to the LRD of A.")
                st.markdown(
                    "Intuitively, if the point is not an outlier (inlier), the ratio of average LRD of neighbors is approximately equal to the LRD of a point (because the density of a point and its neighbors are roughly equal). In that case, LOF is nearly equal to 1. On the other hand, if the point is an outlier, the LRD of a point is less than the average LRD of neighbors. Then LOF value will be high.")
                st.markdown(
                    " :red[Generally, if LOF> 1, it is considered as an outlier], but that is not always true. Let’s say we know that we only have one outlier in the data, then we take the maximum LOF value among all the LOF values, and the point corresponding to the maximum LOF value will be considered as an outlier.")
                col11, col22 = st.columns(2)
                with col11:
                    st.markdown("### :red[**LOF >> 1 (Anomaly)**]")
                with col22:
                    st.markdown("### :green[**LOF ~= 1 (Normal)**]")

                st.subheader("Advantages of LOF")
                st.markdown(
                    "A point will be considered as an outlier if it is at a small distance to the extremely dense cluster. The global approach may not consider that point as an outlier. But the LOF can effectively identify the local outliers.")
                st.subheader("Disavabtages of LOF")
                st.markdown(
                    "Since LOF is a ratio, it is tough to interpret. There is no specific threshold value above which a point is defined as an outlier. The identification of an outlier is dependent on the problem and the user.")


    # -------------------------------------------------------------
    if page_selection == "Step 3: Output":
        # Header Contents
        st.write("# Output of the Model")
        sys = st.radio("**Select an algorithm**", ('Statistical Analysis','Isolation Forest', 'Local Outlier Factor'))

        # Perform top-10 movie recommendation generation
        if sys == 'Statistical Analysis':
            if st.button("Detect"):
                try:
                    with st.spinner("Fitting Benford's Law ..."):
                        st.markdown("#### Analysis on Income Statement")
                        st.markdown("**Construction Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isconrev.png')
                            st.image(image2, caption='Revenue')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iscongro.png')
                            st.image(image2, caption='Gross Revenue')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isconebi.png')
                            st.image(image2, caption='EBIT')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isconnet.png')
                            st.image(image2, caption='NetProfitAfterTax')

                        st.markdown("**UNKNOWN Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isunkrev.png')
                            st.image(image2, caption='Revenue')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isunkgro.png')
                            st.image(image2, caption='Gross Revenue')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isunkebi.png')
                            st.image(image2, caption='EBIT')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isunknet.png')
                            st.image(image2, caption='NetProfitAfterTax')

                        st.markdown("**Logistic Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iclogrev.png')
                            st.image(image2, caption='Revenue')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isloggros.png')
                            st.image(image2, caption='Gross Revenue')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iclogebi.png')
                            st.image(image2, caption='EBIT')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iclognetprof.png')
                            st.image(image2, caption='NetProfitAfterTax')

                        st.markdown("**Energy Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isenerev.png')
                            st.image(image2, caption='Revenue')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isenegro.png')
                            st.image(image2, caption='Gross Revenue')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/iseneebi.png')
                            st.image(image2, caption='EBIT')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/isenenet.png')
                            st.image(image2, caption='NetProfitAfterTax')


                        st.markdown("#### Analysis on Balance Sheet")
                        st.markdown("**Construction Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bstotalequ.png')
                            st.image(image2, caption='Total Equity')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bscontotalass.png')
                            st.image(image2, caption='Total Assets')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bscontotallia.png')
                            st.image(image2, caption='Total Liabilities')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bscontotalnet.png')
                            st.image(image2, caption='Networth')

                        st.markdown("**UNKNOWN Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsunktotalass.png')
                            st.image(image2, caption='Total Assets')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsunktotallia.png')
                            st.image(image2, caption='Total liability')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsunktotalequ.png')
                            st.image(image2, caption='Total Equity')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsunknetw.png')
                            st.image(image2, caption='Networth')

                        st.markdown("**Logistic Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bslogtotalequ.png')
                            st.image(image2, caption='Total Equity')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bslogtotalass.png')
                            st.image(image2, caption='Total Assets')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bslogtotallia.png')
                            st.image(image2, caption='Total Liabilities')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bslognetw.png')
                            st.image(image2, caption='Networth')

                        st.markdown("**Energy Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsenetotalass.png')
                            st.image(image2, caption='Total Assets')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsenetotallia.png')
                            st.image(image2, caption='Total liability')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsenetotalequ.png')
                            st.image(image2, caption='Total Equity')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/bsenenetw.png')
                            st.image(image2, caption='Networth')



                        st.markdown("#### Analysis on Cashflow Statement")
                        st.markdown("**Construction Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfconnetcf.png')
                            st.image(image2, caption='CFF_NetCFF')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfconcash.png')
                            st.image(image2, caption='CFF_CashAtStartOfYear')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfconcashend.png')
                            st.image(image2, caption='CFF_CashAtEndOfYear')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfconnetcfo.png')
                            st.image(image2, caption='CFO_NetCFO')


                        st.markdown("**UNKNOWN Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfunknetcf.png')
                            st.image(image2, caption='CFF_NetCFF')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfunkcash.png')
                            st.image(image2, caption='CFF_CashAtStartOfYear')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfunkcashend.png')
                            st.image(image2, caption='CFF_CashAtEndOfYear')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfunknetcfo.png')
                            st.image(image2, caption='CFO_NetCFO')

                        st.markdown("**Logistic Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cflognetcf.png')
                            st.image(image2, caption='CFF_NetCFF')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cflogcash.png')
                            st.image(image2, caption='CFF_CashAtStartOfYear')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cflogcashend.png')
                            st.image(image2, caption='CFF_CashAtEndOfYear')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cflogcashend.png')
                            st.image(image2, caption='CFO_NetCFO')

                        st.markdown("**Energy Industry**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfenenetcf.png')
                            st.image(image2, caption='CFF_NetCFF')
                        with col2:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfenecash.png')
                            st.image(image2, caption='CFF_CashAtStartOfYear')
                        with col3:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfenecashend.png')
                            st.image(image2, caption='CFF_CashAtEndOfYear')
                        with col4:
                            from PIL import Image
                            image2 = Image.open('resources/imgs/cfenenetcfo.png')
                            st.image(image2, caption='CFO_NetCFO')

                except Exception as e:
                    st.write(e)
                    st.error("Oops! Looks like this algorithm doesn't work.\
                              We'll need to fix it!")


        if sys == 'Isolation Forest':
            if st.button("Detect"):
                try:
                    with st.spinner('Algorithm running...'):
                        df1 = pd.read_csv('resources/data/financials_data.csv')

                        # Convert date column to datetime
                        df1['FinancialsDate'] = pd.to_datetime(df1['FinancialsDate'])
                        df1 = df1.drop('FinancialsDate', axis=1)

                        # Encode categorical columns using one-hot encoding
                        encoder = OneHotEncoder()
                        encoded_cat_columns = encoder.fit_transform(df1[['Financial_Type', 'Country', 'Industry']])
                        encoded_cat_columns_df = pd.DataFrame(encoded_cat_columns.toarray(),
                                                              columns=encoder.get_feature_names(
                                                                  ['Financial_Type', 'Country', 'Industry']))

                        # Combine encoded categorical columns with numerical columns
                        X = pd.concat(
                            [df1.drop(['Financial_Type', 'Country', 'Industry'], axis=1), encoded_cat_columns_df],
                            axis=1)

                        iforest = IsolationForest(max_samples='auto', bootstrap=False, n_jobs=-1, random_state=42)
                        iforest_ = iforest.fit(X)
                        y_pred = iforest_.predict(X)

                        y_score = iforest.decision_function(X)
                        neg_value_indices = np.where(y_score < 0)
                        arr = [ 159,  160,  195,  196,  247,  321,  325,  379,  492,  493,  512,
         532,  533,  534,  607,  608,  776,  790,  941,  943, 1019, 1020,
        1066, 1096, 1196, 1197, 1301, 1311, 1312, 1318, 1319, 1320, 1321,
        1518, 1588, 1606, 1607, 1752, 1753, 1767, 1843, 1844, 1852, 1853,
        1854, 1855, 1857, 1858, 1869, 1870, 1871, 1872, 1923, 1924, 1928,
        2010, 2020, 2022, 2023, 2026, 2027, 2028, 2029, 2140, 2141, 2142,
        2143, 2187, 2188, 2221, 2222, 2226, 2227, 2239, 2240, 2247, 2336,
        2337, 2341, 2342, 2350, 2351, 2354, 2355, 2356, 2357, 2382, 2383,
        2428, 2429, 2492, 2493, 2494, 2495, 2496, 2497, 2499, 2518, 2522,
        2523, 2524, 2525, 2533, 2540, 2555, 2556, 2557, 2564, 2565, 2572,
        2578, 2579, 2602, 2606, 2611, 2623, 2626, 2632, 2635, 2636, 2638,
        2639, 2640, 2641, 2642, 2677, 2680, 2695, 2696, 2698, 2699, 2720,
        2730, 2731, 2750, 2751, 2752, 2753, 2757, 2758, 2763, 2795, 2796,
        2805, 2811, 2812, 2813, 2818, 2820, 2828, 2829, 2854, 2855, 2860,
        2862, 2863, 2868, 2886, 2887, 2888, 2889, 2890, 2891, 2893, 2894,
        2895, 2896, 2897, 2898, 2903, 2904, 2905, 2931, 2932, 2933, 2939,
        2940, 2941, 2942, 2943, 2944, 2947, 2952, 2961, 2962, 2987, 2990,
        2991, 2992, 2993, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3008,
        3019, 3047, 3048, 3050, 3051, 3052, 3060, 3061, 3064, 3066, 3067,
        3081, 3082, 3093, 3096, 3097, 3098, 3099, 3100, 3101, 3109, 3126,
        3127, 3128, 3137, 3156, 3159, 3172, 3182, 3207, 3208, 3209, 3210,
        3212, 3213, 3268, 3270, 3271, 3286, 3294, 3295, 3296, 3436, 3437,
        3438, 3451, 3452, 3453, 3462, 3463, 3464, 3493, 3568, 3569, 3603,
        3604, 3605, 3606, 3607, 3610, 3613, 3614, 3619, 3620, 3621, 3622,
        3623, 3626, 3629, 3633, 3634, 3635, 3641, 3642, 3643, 3644, 3645,
        3651, 3652, 3653, 3654, 3655, 3656, 3659, 3662, 3666, 3667, 3668,
        3669, 3670, 3671, 3672, 3679, 3680, 3681, 3688, 3689, 3690, 3691,
        3692, 3694, 3695, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720,
        3721, 3722, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3759, 3760,
        3761, 3762, 3767, 3770, 3771, 3776, 3777, 3778, 3779, 3780, 3781,
        3782, 3783, 3784, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796,
        3797, 3805, 3806, 3811, 3822, 3823, 3942, 3968, 3969, 3985, 3986,
        4016, 4032, 4037, 4038, 4066, 4069, 4070, 4075, 4076, 4077, 4103,
        4104, 4114, 4115, 4116, 4127, 4132, 4133, 4136, 4138, 4139, 4145,
        4147, 4150, 4151, 4154, 4155, 4158, 4159, 4163, 4164, 4165, 4166,
        4167, 4168, 4169, 4170, 4171, 4172, 4174, 4175, 4184, 4185, 4192,
        4201, 4202, 4208, 4214, 4215, 4216, 4229, 4230, 4231, 4241, 4249,
        4250, 4282, 4283, 4290, 4303, 4304, 4307, 4313, 4314, 4315, 4316,
        4317, 4318, 4322, 4333, 4334, 4335, 4336, 4338, 4359, 4378, 4379,
        4380, 4381, 4382, 4400, 4405, 4412, 4413, 4420, 4421, 4434, 4435,
        4436, 4452, 4453, 4454, 4480, 4482, 4500, 4501, 4509, 4513, 4568]

                        # Converting that array into a dataframe
                        outliers_df = df1.iloc[arr]
                        st.success("### Results are ready!")
                        st.markdown("### 451 possible fraudulant statements identified")
                        st.dataframe(outliers_df)  # Same as st.write(df)
                        # model here

                except Exception as e:
                    st.write(e)
                    st.error("Oops! Looks like this algorithm doesn't work.\
                                      We'll need to fix it!")

                df_final = outliers_df
        st.subheader('Isolation Forest Interpretability')
        col1, col2 = st.columns(2)
        if st.checkbox("Global Interpretability"):
            st.subheader("Global Machine Learning Interpretability")
            st.image('resources/imgs/ifglobal.png',use_column_width=True, caption = 'Summary Plot.')
            st.markdown("From this plot, the impact of a particular variable on anomaly detection is observed. Taking NCA_TotalLoansIssued, CL_InstalmentSaleLiabilty or CL_BankOverdraft as an example. The summary plot says that high values of that variables show anomalous observations while lower values are normal items.")

            st.image('resources/imgs/ifglobalbar.png', use_column_width=True, caption='Bar Plot.')
            st.markdown(
            "From the above, the variables CA_TradeAndOtherRecievables, NCA_TotalLoansIssued and VA_NCL_TotalEquityAndLiability_TotalEquity have the highest average SHAP value. Hence, they have the highest impact on determining the anomaly score..")

        if st.checkbox("Local Interpretability"):
            st.subheader("Local Machine Learning Interpretability")

            my_list = [159,  160,  195,  196,  247,  321,  325,  379,  492,  493,  512,
         532,  533,  534,  607,  608,  776,  790,  941,  943, 1019, 1020,
        1066, 1096, 1196, 1197, 1301, 1311, 1312, 1318, 1319, 1320, 1321,
        1518, 1588, 1606, 1607, 1752, 1753, 1767, 1843, 1844, 1852, 1853,
        1854, 1855, 1857, 1858, 1869, 1870, 1871, 1872, 1923, 1924, 1928,
        2010, 2020, 2022, 2023, 2026, 2027, 2028, 2029, 2140, 2141, 2142,
        2143, 2187, 2188, 2221, 2222, 2226, 2227, 2239, 2240, 2247, 2336,
        2337, 2341, 2342, 2350, 2351, 2354, 2355, 2356, 2357, 2382, 2383,
        2428, 2429, 2492, 2493, 2494, 2495, 2496, 2497, 2499, 2518, 2522,
        2523, 2524, 2525, 2533, 2540, 2555, 2556, 2557, 2564, 2565, 2572,
        2578, 2579, 2602, 2606, 2611, 2623, 2626, 2632, 2635, 2636, 2638,
        2639, 2640, 2641, 2642, 2677, 2680, 2695, 2696, 2698, 2699, 2720,
        2730, 2731, 2750, 2751, 2752, 2753, 2757, 2758, 2763, 2795, 2796,
        2805, 2811, 2812, 2813, 2818, 2820, 2828, 2829, 2854, 2855, 2860,
        2862, 2863, 2868, 2886, 2887, 2888, 2889, 2890, 2891, 2893, 2894,
        2895, 2896, 2897, 2898, 2903, 2904, 2905, 2931, 2932, 2933, 2939,
        2940, 2941, 2942, 2943, 2944, 2947, 2952, 2961, 2962, 2987, 2990,
        2991, 2992, 2993, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3008,
        3019, 3047, 3048, 3050, 3051, 3052, 3060, 3061, 3064, 3066, 3067,
        3081, 3082, 3093, 3096, 3097, 3098, 3099, 3100, 3101, 3109, 3126,
        3127, 3128, 3137, 3156, 3159, 3172, 3182, 3207, 3208, 3209, 3210,
        3212, 3213, 3268, 3270, 3271, 3286, 3294, 3295, 3296, 3436, 3437,
        3438, 3451, 3452, 3453, 3462, 3463, 3464, 3493, 3568, 3569, 3603,
        3604, 3605, 3606, 3607, 3610, 3613, 3614, 3619, 3620, 3621, 3622,
        3623, 3626, 3629, 3633, 3634, 3635, 3641, 3642, 3643, 3644, 3645,
        3651, 3652, 3653, 3654, 3655, 3656, 3659, 3662, 3666, 3667, 3668,
        3669, 3670, 3671, 3672, 3679, 3680, 3681, 3688, 3689, 3690, 3691,
        3692, 3694, 3695, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720,
        3721, 3722, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3759, 3760,
        3761, 3762, 3767, 3770, 3771, 3776, 3777, 3778, 3779, 3780, 3781,
        3782, 3783, 3784, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796,
        3797, 3805, 3806, 3811, 3822, 3823, 3942, 3968, 3969, 3985, 3986,
        4016, 4032, 4037, 4038, 4066, 4069, 4070, 4075, 4076, 4077, 4103,
        4104, 4114, 4115, 4116, 4127, 4132, 4133, 4136, 4138, 4139, 4145,
        4147, 4150, 4151, 4154, 4155, 4158, 4159, 4163, 4164, 4165, 4166,
        4167, 4168, 4169, 4170, 4171, 4172, 4174, 4175, 4184, 4185, 4192,
        4201, 4202, 4208, 4214, 4215, 4216, 4229, 4230, 4231, 4241, 4249,
        4250, 4282, 4283, 4290, 4303, 4304, 4307, 4313, 4314, 4315, 4316,
        4317, 4318, 4322, 4333, 4334, 4335, 4336, 4338, 4359, 4378, 4379,
        4380, 4381, 4382, 4400, 4405, 4412, 4413, 4420, 4421, 4434, 4435,
        4436, 4452, 4453, 4454, 4480, 4482, 4500, 4501, 4509, 4513, 4568]

            option = st.selectbox(
                'Select a financial statement', my_list)
            if option == 159:
                st.image('resources/imgs/159.png', use_column_width=True,
                         caption='Anomaly 159')
            if option == 160:
                st.image('resources/imgs/160.png', use_column_width=True,
                         caption='Anomaly 160')
            if option == 195:
                st.image('resources/imgs/195.png', use_column_width=True,
                         caption='Anomaly 195')
            if option == 196:
                st.image('resources/imgs/196.png', use_column_width=True,
                         caption='Anomaly 196')
            if option == 2010:
                st.image('resources/imgs/2010.png', use_column_width=True,
                         caption='Anomaly 2010')
            if option == 2350:
                st.image('resources/imgs/2350.png', use_column_width=True,
                         caption='Anomaly 2350')
            if option == 4436:
                st.image('resources/imgs/4436.png', use_column_width=True,
                         caption='Anomaly 4436')
            if option == 4452:
                st.image('resources/imgs/4452.png', use_column_width=True,
                         caption='Anomaly 4450')
            if option == 4453:
                st.image('resources/imgs/4453.png', use_column_width=True,
                         caption='Anomaly 4453')
            if option == 4454:
                st.image('resources/imgs/4454.png', use_column_width=True,
                         caption='Anomaly 4454')
            if option == 4480:
                st.image('resources/imgs/4480.png', use_column_width=True,
                         caption='Anomaly 4480')
            if option == 4482:
                st.image('resources/imgs/4482.png', use_column_width=True,
                         caption='Anomaly 4482')
            if option == 4500:
                st.image('resources/imgs/4500.png', use_column_width=True,
                         caption='Anomaly 4500')
            if option == 4568:
                st.image('resources/imgs/4568.png', use_column_width=True,
                         caption='Anomaly 4568')
#---------------------------------------------------------
            if option == 2020:
                st.image('resources/imgs/2020.png', use_column_width=True,
                         caption='Anomaly 2020')
            if option == 2023:
                st.image('resources/imgs/2023.png', use_column_width=True,
                         caption='Anomaly 2023')
            if option == 2026:
                st.image('resources/imgs/2026.png', use_column_width=True,
                         caption='Anomaly 2026')
            if option == 3000:
                st.image('resources/imgs/3000.png', use_column_width=True,
                         caption='Anomaly 3000')
            if option == 3050:
                st.image('resources/imgs/3050.png', use_column_width=True,
                         caption='Anomaly 3050')
            if option == 3666:
                st.image('resources/imgs/3666.png', use_column_width=True,
                         caption='Anomaly 3666')
            if option == 2141:
                st.image('resources/imgs/2141.png', use_column_width=True,
                         caption='Anomaly 2141')




    if page_selection == "Step 1: Input":
        df = pd.read_csv('resources/data/financials_data.csv')
        st.title('Input')
        st.markdown("In this section, the user can view the data that was used as an input to the model. Input data are are he financial statements that will be used for analysis. The financial statements include, balance sheets, income statements, and cash flow statements, "
                    "in a specific format. The app validates the data entered by the user and provide feedback in case of any errors or inconsistencies")

        st.warning('The graphs that appear are interactive and can be zoomed in or certain data can be selected')

        if st.checkbox("Show raw data"):
            st.subheader("Financial Statements")
            st.dataframe(df)  # Same as st.write(df)
            st.subheader("Summary statistics of the dataset")
            df1 =df.describe()
            st.table(df1)


        if st.checkbox("Show explanation"):
            st.subheader("The input dataset")
            st.markdown("The data consists of 1,000s of financial statements (i.e. income statement, balance sheet, cash flow statement, financial type, year), details of the company (industry, age) and "
                        "whether the company defaulted in the 12-months post the date of the financial statement.")

            st.markdown("- There are ~8,000 rows (i.e. financial statements) and ~100 columns (financial entries, company details and default indicator")
            st.markdown("- There is a time component, which is one dimension we want to analyse the predicted fraud by")

            st.subheader("Exploratory Data Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.image('resources/imgs/financial_type.png',use_column_width=True, caption= 'The plot shows the frequencies of the different financial types')

            with col2:
                st.image('resources/imgs/industry.png',use_column_width=True, caption= 'The plot shows the frequencies of the different industry types')


            st.markdown('The most frequent financial types are Audited - Signed (3341) and Financials - By Accounting Officer - Signed (916). These two make up more than 90% of dataset')
            st.markdown("Checking through the Industry type, the most frequent industry types are Construction (2030) and Unknown (1759). The industry named 0 will be added to the UNKNOWN industry as it is also an unknown industry")

            st.subheader("Region of financial statements")
            st.image('resources/imgs/country.png',use_column_width=True, caption='The most frequent countries are South Africa (2663) and UNKNOWN (1762) which is more than 90%. The Country named AFRICA will be captured in UNKNOWN country as there is no country named AFRICA')

            st.subheader("Dates of financial statements")
            st.image('resources/imgs/dates_percentage.png', caption='From the plot, it can be deduced that the financial statements in not evenly distributed across the years, There are 11% (highest value) financial year end in 2017 February followed by 9% in 2018, February. Others that presented in January, March - November makes up 38.8% of the distribution.')

            st.subheader("Plot the proportion of default in dataset")
            st.image('resources/imgs/default.png')
            st.markdown("Only 4.4% (203) transactions in the dataset are default while 95.6% (4382) transactions are nondefault with ratio of 21.59 indicating hight class imbalance in the dataset. Building a machine learning model on a highly skewed data as shown here, the nondefault transactions will influence the training of the model almost entirely, thus affecting the results.")

if __name__ == '__main__':
    main()
