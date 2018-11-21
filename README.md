These are the codes for paper Aesthetic-based Clothing Recommedation
We also provide a small dataset filtered for Amazon dataset (Jewelry and watches) to run the codes. There are 15924 users, 3607 items, and 37314 purchases in the dataset.
url for dataset: https://pan.baidu.com/s/1xIFdR8lpd5mui-baTIM_xA

The forms of data in each file are:
1. interactions_Jewelry_train: [[u,i,r],...,[u,i,r]]
a list storing training samples, where u, i, r are user, item, time respectively.
2. interactions_Jewelry_test: [[u,[i,i,i],[r,r]],...,[[u,[i,i],[r,r]]]
a list storing test samples, each element of the list ([u,[i,i,i],[r,r]]) consists of a user u, the set of purchased items [i,i,i], and the set of time [r,r]. 
3. interactions_Jewelry_train_aux: [[[i,i],[r]],...,[[i,i,i],[r,r]]]
a list storing training samples, where the u-th elemet [[i,i],[r]] is the set of purchase items and time of user u.
4. interactions_Jewelry_train_record_aux: [[[i,r]],...,[[i,r],[i,r],[i,r]]]
a list storing training samples, where the u-th elemet [[i,r],[i,r],[i,r]] is the set of purchased item-time pairs of user u. In each pair, i is the purchases item and r is the corresponding time.
5. interactions_Jewelry_train_time_aux: [[[u,u,u,u],[i,i,i]],...,[[u,u],[i,i,i]]]
a list storing training samples, where the r-th elemet [[u,u],[i,i,i]] is the set of users who purchased something in time r and the set of items purchsed in r.
6. interactions_Jewelry_validate: [[u,[i,i,i],[r,r]],...,[[u,[i,i],[r,r]]]
a list storing validation samples, each element of the list ([u,[i,i,i],[r,r]]) consists of a user u, the set of purchased items [i,i,i], and the set of time [r,r]. 

The feature provided is concatenation of CNN feature and aesthetic feature, we then reduce its dimention with PCA