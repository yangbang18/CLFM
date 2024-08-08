# root paths to load raw images or videos
image_video_root = {
    'coco': 'data/MSCOCO',
    'flickr30k': 'data/Flickr30k',
    'msrvtt': 'data/MSRVTT',
    'vatex': 'data/VATEX',
    'xm3600': 'data/XM3600',
    'cc3m': 'data/cc3m',
}

num_frames = 8 # the number of frames to be uniformly sampled for each video

# path to store train/val/test splits of each dataset
annotation_root = 'data/annotations' 
# path to store files solely for training
corpus_root = 'data/corpus' 
tokenizer_root = 'data/tokenizers'

# generation settings for visual captioning in different languages
auto_settings = {
    'en': dict(
        max_length=20,
        min_length=3,
        repetition_penalty=1.0,
    ),
    'zh': dict(
        max_length=30,
        min_length=3,
        repetition_penalty=1.0,
    ),
    'de': dict(
        max_length=15,
        min_length=3,
        repetition_penalty=2.0,
    ),
    'fr': dict(
        max_length=20,
        min_length=3,
        repetition_penalty=2.0,
    ),
}

flickr30k_langs = ['en', 'de', 'fr', 'cs']
flickr30k_order1 = ['de', 'fr', 'cs', 'en']

xm3600_langs = ['ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fil', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'quz', 'ro', 'ru', 'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh']
xm3600_order1 = xm3600_langs
xm3600_seed222 = ['it', 'de', 'fil', 'tr', 'uk', 'nl', 'bn', 'he', 'fi', 'sv', 'quz', 'fa', 'mi', 'fr', 'el', 'zh', 'id', 'sw', 'no', 'hi', 'da', 'te', 'th', 'pt', 'ru', 'ro', 'pl', 'hr', 'es', 'vi', 'ar', 'cs', 'ko', 'ja', 'hu', 'en']
xm3600_order222 = ['en', 'it', 'de', 'fil', 'tr', 'uk', 'nl', 'bn', 'he', 'fi', 'sv', 'quz', 'fa', 'mi', 'fr', 'el', 'zh', 'id', 'sw', 'no', 'hi', 'da', 'te', 'th', 'pt', 'ru', 'ro', 'pl', 'hr', 'es', 'vi', 'ar', 'cs', 'ko', 'ja', 'hu']


language_info = {
    'ar': dict(
        name='Arabic',
        script='Arabic',
        family='Afro-Asiatic',
        branch='',
        nllb_special_token='', #TODO, https://zhuanlan.zhihu.com/p/667257196
    ),
    'bn': dict(
        name='Bengali',
        script='Bengali',
        family='Indo-European',
        branch='Indo-Iranian',
        nllb_special_token='ben_Beng', 
    ),
    'cs': dict(
        name='Czech',
        script='Latin',
        family='Indo-European',
        branch='Balto-Slavic',
        nllb_special_token='ces_Latn',
    ),
    'da': dict(
        name='Danish',
        script='Latin',
        family='Indo-European',
        branch='North Germanic',
        nllb_special_token='dan_Latn',
    ),
    'de': dict(
        name='German',
        script='Latin',
        family='Indo-European',
        branch='West Germanic',
        nllb_special_token='deu_Latn',
    ),
    'el': dict(
        name='Greek',
        script='Latin',
        family='Indo-European',
        branch='Hellenic',
        nllb_special_token='ell_Grek',
    ),
    'en': dict(
        name='English',
        script='Latin',
        family='Indo-European',
        branch='West Germanic',
        nllb_special_token='eng_Latn',
    ),
    'es': dict(
        name='Spanish',
        script='Latin',
        family='Indo-European',
        branch='Italic',
        nllb_special_token='spa_Latn',
    ),
    'fa': dict(
        name='Persian',
        script='Arabic',
        family='Indo-European',
        branch='Indo-Iranian',
        nllb_special_token='pes_Arab',
    ),
    'fi': dict(
        name='Finnish',
        script='Latin',
        family='Uralic',
        branch='Finnic',
        nllb_special_token='fin_Latn',
    ),
    'fil': dict(
        name='Filipino',
        script='Latin',
        family='Austronesian',
        branch='Malayo-Polynesian',
        nllb_special_token='', #TODO
    ),
    'fr': dict(
        name='French',
        script='Latin',
        family='Indo-European',
        branch='Italic',
        nllb_special_token='fra_Latn',
    ),
    'he': dict(
        name='Hebrew',
        script='Hebrew',
        family='Afro-Asiatic',
        branch='Semitic',
        nllb_special_token='heb_Hebr',
    ),
    'hi': dict(
        name='Hindi',
        script='Devanagari',
        family='Indo-European',
        branch='Indo-Iranian',
        nllb_special_token='hin_Deva',
    ),
    'hr': dict(
        name='Croatian',
        script='Latin',
        family='Indo-European',
        branch='Balto-Slavic',
        nllb_special_token='hrv_Latn',
    ),
    'hu': dict(
        name='Hungarian',
        script='Latin',
        family='Uralic',
        branch='',
        nllb_special_token='hun_Latn',
    ),
    'id': dict(
        name='Indonesian',
        script='Latin',
        family='Austronesian',
        branch='Malayo-Polynesian',
        nllb_special_token='ind_Latn',
    ),
    'it': dict(
        name='Italian',
        script='Latin',
        family='Indo-European',
        branch='Italic',
        nllb_special_token='ita_Latn',
    ),
    'ja': dict(
        name='Japanese',
        script='Kanji',
        family='Japonic',
        branch='',
        nllb_special_token='jpn_Jpan',
    ),
    'ko': dict(
        name='Korean',
        script='Hangul',
        family='Koreanic',
        branch='',
        nllb_special_token='kor_Hang',
    ),
    'mi': dict(
        name='Māori',
        script='Latin',
        family='Austronesian',
        branch='Malayo-Polynesian',
        nllb_special_token='mri_Latn',
    ),
    'nl': dict(
        name='Dutch',
        script='Latin',
        family='Indo-European',
        branch='West Germanic',
        nllb_special_token='nld_Latn',
    ),
    'no': dict(
        name='Norwegian',
        script='Latin',
        family='Indo-European',
        branch='North Germanic',
        nllb_special_token='nno_Latn',
    ),
    'pl': dict(
        name='Polish',
        script='Latin',
        family='Indo-European',
        branch='Balto-Slavic',
        nllb_special_token='pol_Latn',
    ),
    'pt': dict(
        name='Portuguese',
        script='Latin',
        family='Indo-European',
        branch='Italic',
        nllb_special_token='por_Latn',
    ),
    'quz': dict(
        name='Cuzco Quechua',
        script='Latin',
        family='Quechuan',
        branch='',
        nllb_special_token='quy_Latn',
    ),
    'ro': dict(
        name='Romanian',
        script='Latin',
        family='Indo-European',
        branch='Italic',
        nllb_special_token='ron_Latn',
    ),
    'ru': dict(
        name='Russian',
        script='Cyrillic',
        family='Indo-European',
        branch='Balto-Slavic',
        nllb_special_token='rus_Cyrl',
    ),
    'sv': dict(
        name='Swedish',
        script='Latin',
        family='Indo-European',
        branch='North Germanic',
        nllb_special_token='swe_Latn',
    ),
    'sw': dict(
        name='Swahili',
        script='Latin',
        family='Niger–Congo',
        branch='',
        nllb_special_token='swh_Latn',
    ),
    'te': dict(
        name='Telugu',
        script='Telugu',
        family='Dravidian',
        branch='South-Central',
        nllb_special_token='tel_Telu',
    ),
    'th': dict(
        name='Thai',
        script='Thai',
        family='Kra-Dai',
        branch='Tai',
        nllb_special_token='tha_Thai',
    ),
    'tr': dict(
        name='Turkish',
        script='Latin',
        family='Turkic',
        branch='',
        nllb_special_token='tur_Latn',
    ),
    'uk': dict(
        name='Ukrainian',
        script='Cyrillic',
        family='Indo-European',
        branch='Balto-Slavic',
        nllb_special_token='ukr_Cyrl',
    ),
    'vi': dict(
        name='Vietnamese',
        script='Latin',
        family='Austroasiatic',
        branch='',
        nllb_special_token='vie_Latn',
    ),
    'zh': dict(
        name='Chinese',
        script='Chinese Characters',
        family='Sino-Tibetan',
        branch='Sinitic',
        nllb_special_token='zho_Hans',
    ),
}
