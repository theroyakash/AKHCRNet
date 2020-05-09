# AKHCRNetV1 (Research)
Deep neural architecture on bengali hand written character.

Here is the model code. (At least working okay)
```Python
# Model architecture (Working)
img_input = Input(shape=(200,200,3))

conv2d_1 = Conv2D(64, (3,3), activation='relu', padding='valid', name='conv2d_1')(img_input)
maxpool1 = MaxPooling2D(pool_size=(2,2))(conv2d_1)

conv2d_2 = Conv2D(128, (3,3), activation='relu', padding='valid', name='conv2d_2')(maxpool1)

conv2d_3 = Conv2D(128, (3,3), activation='relu', padding='valid', name='conv2d_3')(conv2d_2)
maxpool1 = MaxPooling2D(pool_size=(2,2))(conv2d_3)

conv2d_5 = Conv2D(128, (3,3), activation='relu', padding='valid', name='conv2d_5')(maxpool1)

branch0 = Conv2D(64, (1,1), padding='same', name='Branch_Zero_1_by_1_Conv2D')(conv2d_5)

branch1 = Conv2D(64, (1,1), activation='relu', padding='same', name='BranchOne3By3Conv2D1')(conv2d_5)
branch1 = Conv2D(64, (3,3), activation='relu', padding='same', name='BranchOne3By3Conv2D2')(branch1)
branch1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='BranchOne3By3Conv2D3')(branch1)

concatenated_branchA = Concatenate()([branch0, branch1])

pool0 = MaxPooling2D(pool_size=(2, 2))(concatenated_branchA)

branch00 = Conv2D(64, (1,1), padding='same', name='BranchZeroZero1By1Conv2D')(pool0)

branch11 = Conv2D(64, (1,1), activation='relu', padding='same', name='BranchOneOne3By3Conv2D1')(pool0)
branch11 = Conv2D(64, (3,3), activation='relu', padding='same', name='BranchOneOne3By3Conv2D2')(branch11)
branch11 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='BranchOneOne3By3Conv2D3')(branch11)

concatenated_branchB = Concatenate()([branch00, branch11])

flattened_before_dense = Flatten()(concatenated_branchB)
dense1 = Dense(1024, activation='relu', name='firstDenseLayer')(flattened_before_dense)
dense2 = Dense(512, activation='relu', name='SecondDenseLayer')(dense1)
prediction_branch = Dense(84,activation='softmax', name='FinalSoftmaxLayer')(dense2)

model = Model(inputs=img_input, outputs=prediction_branch)
model.summary()
```

# Output
```Plain Text
Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_7 (InputLayer)            [(None, 200, 200, 3) 0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 198, 198, 64) 1792        input_7[0][0]                    
__________________________________________________________________________________________________
max_pooling2d_16 (MaxPooling2D) (None, 99, 99, 64)   0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 97, 97, 128)  73856       max_pooling2d_16[0][0]           
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 95, 95, 128)  147584      conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_17 (MaxPooling2D) (None, 47, 47, 128)  0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 45, 45, 128)  147584      max_pooling2d_17[0][0]           
__________________________________________________________________________________________________
BranchOne3By3Conv2D1 (Conv2D)   (None, 45, 45, 64)   8256        conv2d_5[0][0]                   
__________________________________________________________________________________________________
BranchOne3By3Conv2D2 (Conv2D)   (None, 45, 45, 64)   36928       BranchOne3By3Conv2D1[0][0]       
__________________________________________________________________________________________________
Branch_Zero_1_by_1_Conv2D (Conv (None, 45, 45, 64)   8256        conv2d_5[0][0]                   
__________________________________________________________________________________________________
BranchOne3By3Conv2D3 (Conv2D)   (None, 45, 45, 32)   18464       BranchOne3By3Conv2D2[0][0]       
__________________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, 45, 45, 96)   0           Branch_Zero_1_by_1_Conv2D[0][0]  
                                                                 BranchOne3By3Conv2D3[0][0]       
__________________________________________________________________________________________________
max_pooling2d_18 (MaxPooling2D) (None, 22, 22, 96)   0           concatenate_10[0][0]             
__________________________________________________________________________________________________
BranchOneOne3By3Conv2D1 (Conv2D (None, 22, 22, 64)   6208        max_pooling2d_18[0][0]           
__________________________________________________________________________________________________
BranchOneOne3By3Conv2D2 (Conv2D (None, 22, 22, 64)   36928       BranchOneOne3By3Conv2D1[0][0]    
__________________________________________________________________________________________________
BranchZeroZero1By1Conv2D (Conv2 (None, 22, 22, 64)   6208        max_pooling2d_18[0][0]           
__________________________________________________________________________________________________
BranchOneOne3By3Conv2D3 (Conv2D (None, 22, 22, 32)   18464       BranchOneOne3By3Conv2D2[0][0]    
__________________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, 22, 22, 96)   0           BranchZeroZero1By1Conv2D[0][0]   
                                                                 BranchOneOne3By3Conv2D3[0][0]    
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 46464)        0           concatenate_11[0][0]             
__________________________________________________________________________________________________
firstDenseLayer (Dense)         (None, 1024)         47580160    flatten_5[0][0]                  
__________________________________________________________________________________________________
SecondDenseLayer (Dense)        (None, 512)          524800      firstDenseLayer[0][0]            
__________________________________________________________________________________________________
FinalSoftmaxLayer (Dense)       (None, 84)           43092       SecondDenseLayer[0][0]           
==================================================================================================
Total params: 48,658,580
Trainable params: 48,658,580
Non-trainable params: 0
__________________________________________________________________________________________________
```
