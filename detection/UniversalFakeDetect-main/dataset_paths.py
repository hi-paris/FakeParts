DATASET_PATHS = [


    dict(
        real_path='./datasets/test/progan',     
        fake_path='./datasets/test/progan',
        data_mode='wang2020',
        key='progan'
    ),

    dict(
        real_path='./datasets/test/cyclegan',   
        fake_path='./datasets/test/cyclegan',
        data_mode='wang2020',
        key='cyclegan'
    ),

    dict(
        real_path='./datasets/test/biggan/',   # Imagenet 
        fake_path='./datasets/test/biggan/',
        data_mode='wang2020',
        key='biggan'
    ),


    dict(
        real_path='./datasets/test/stylegan',    
        fake_path='./datasets/test/stylegan',
        data_mode='wang2020',
        key='stylegan'
    ),

    dict(
        real_path='./datasets/test/stylegan2',    
        fake_path='./datasets/test/stylegan2',
        data_mode='wang2020',
        key='stylegan2'
    ),

    dict(
        real_path='./datasets/test/whichfaceisreal',    
        fake_path='./datasets/test/whichfaceisreal',
        data_mode='wang2020',
        key='whichfaceisreal'
    ),


    dict(
        real_path='./datasets/test/gaugan',    # It is COCO 
        fake_path='./datasets/test/gaugan',
        data_mode='wang2020',
        key='gaugan'
    ),


    dict(
        real_path='./datasets/test/stargan',  
        fake_path='./datasets/test/stargan',
        data_mode='wang2020',
        key='stargan'
    ),


    dict(
        real_path='./datasets/test/deepfake',   
        fake_path='./datasets/test/deepfake',
        data_mode='wang2020',
        key='deepfake'
    ),


    dict(
        real_path='./datasets/test/seeingdark',   
        fake_path='./datasets/test/seeingdark',
        data_mode='wang2020',
        key='seeingdark' #modification
    ),


    dict(
        real_path='./datasets/test/san',   
        fake_path='./datasets/test/san',
        data_mode='wang2020',
        key='san'
    ),


    dict(
        real_path='./datasets/test/crn',   # Images from some video games
        fake_path='./datasets/test/crn',
        data_mode='wang2020',
        key='crn'
    ),


    dict(
        real_path='./datasets/test/imle',   # Images from some video games
        fake_path='./datasets/test/imle',
        data_mode='wang2020',
        key='imle'
    ),
    

    dict(
        real_path='./diffusion_datasets/imagenet',
        fake_path='./diffusion_datasets/guided',
        data_mode='wang2020',
        key='guided'
    ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/ldm_200',
        data_mode='wang2020',
        key='ldm_200'
    ),

    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/ldm_200_cfg',
        data_mode='wang2020',
        key='ldm_200_cfg'
    ),

    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/ldm_100',
        data_mode='wang2020',
        key='ldm_100'
     ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/glide_100_27',
        data_mode='wang2020',
        key='glide_100_27'
    ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/glide_50_27',
        data_mode='wang2020',
        key='glide_50_27'
    ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/glide_100_10',
        data_mode='wang2020',
        key='glide_100_10'
    ),


    dict(
        real_path='./diffusion_datasets/laion',
        fake_path='./diffusion_datasets/dalle',
        data_mode='wang2020',
        key='dalle'
    ),



]
