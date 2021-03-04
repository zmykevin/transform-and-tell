import sys
sys.path.append(r'/home/zmykevin/semafor/code/transform-and-tell')
import os
import torch
import logging
import random
import json
import re
import numpy as np
from PIL import Image

#load from the source code
from news_caption_analytic.commands.train import yaml_to_params

#load from allennlp
from allennlp.common.util import prepare_environment
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.nn.util import move_to_device


# from news_caption_analytic.data.vocabulary import RobertaVocabulary
from news_caption_analytic.models.transformer_faces_objects import TransformerFacesObjectModel
from news_caption_analytic.models.decoder_faces_objects import DynamicConvFacesObjectsDecoder
from news_caption_analytic.facenet import MTCNN, InceptionResnetV1
from news_caption_analytic.models.resnet import resnet152
from news_caption_analytic.yolov3.models import Darknet, attempt_download
from news_caption_analytic.yolov3.utils.utils import (load_classes, non_max_suppression,
                                     plot_one_box, scale_coords)
from news_caption_analytic.yolov3.utils.datasets import letterbox
from news_caption_analytic.data.token_indexers import RobertaTokenIndexer
from news_caption_analytic.data.fields import ImageField

#load from the torchvision
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)

logger = logging.getLogger(__name__)
SPACE_NORMALIZER = re.compile(r"\s+")
ENV = os.environ.copy()

def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def get_obj_embeddings(xyxy, pil_image, obj_path, resnet):
    pil_image = pil_image.convert('RGB')
    obj_image = extract_object(pil_image, xyxy, save_path=obj_path)

    preprocess = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    obj_image = preprocess(obj_image)
    # obj_image.shape == [n_channels, height, width]

    # Add a batch dimension
    obj_image = obj_image.unsqueeze(0).to(next(resnet.parameters()).device)
    # obj_image.shape == [1, n_channels, height, width]
    with torch.no_grad():
	    X_image = resnet(obj_image, pool=True)
	    # X_image.shape == [1, 2048]

	    X_image = X_image.squeeze(0).cpu().numpy().tolist()
	    # X_image.shape == [2048]

    return X_image

def extract_object(img, box, image_size=224, margin=0, save_path=None):
    """Extract object + margin from PIL Image given bounding box.
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted object image. (default: {None})
    Returns:
        torch.tensor -- tensor representing the extracted object.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin)
    ]
    box = [
        int(max(box[0] - margin[0]/2, 0)),
        int(max(box[1] - margin[1]/2, 0)),
        int(min(box[2] + margin[0]/2, img.size[0])),
        int(min(box[3] + margin[1]/2, img.size[1]))
    ]

    obj = img.crop(box).resize((image_size, image_size), 2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path)+'/', exist_ok=True)
        save_args = {'compress_level': 0} if '.png' in save_path else {}
        obj.save(save_path, **save_args)

    return obj

class Captioner(object):
    def __init__(self):
        self.model = None
        self.bpe = None
        self.indices = None
        self.preprocess = None
        self.data_iterator = None
        self.tokenizer = None
        self.token_indexers = None
        self.mtcnn = None
        self.inception = None
        self.resnet = None
        self.darknet = None
        self.names = None
        self.colors = None
        self.nlp = None
        if torch.cuda.is_available():
            #n_devices = torch.cuda.device_count()
            d = 0
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                devs = ENV['CUDA_VISIBLE_DEVICES'].split(',')
                os.environ['CUDA_VISIBLE_DEVICES'] = devs[d]
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(d)
            self.device = torch.device(f'cuda:0')
        else:
            self.device = torch.device('cpu')
        self.config_path = "expt/goodnews/9_transformer_objects/config_caption_generation.yaml"
        self.news_data_path = "data/semafor"
    def initialize(self):
        logger.info(f'loading config from {self.config_path}')
        print(f'loading config from {self.config_path}')
        config = yaml_to_params(self.config_path, overrides='')
        prepare_environment(config)
        #load the vocab
        vocab = Vocabulary.from_params(config.pop('vocabulary'))
        #load the model
        model = Model.from_params(vocab=vocab, params=config.pop('model'))
        model = model.eval()
        model_path = 'expt/goodnews/9_transformer_objects/serialization/best.th'
        best_model_state = torch.load(
            model_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_model_state)

        self.model = model.to(self.device)
        
        logger.info(f'Loading best model from {model_path}')
        print(f'Loading best model from {model_path}')
        roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices
        
        logger.info('Loading face detection model.')
        print('Loading face detection model')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.inception = InceptionResnetV1(pretrained='vggface2').eval()

        self.resnet = resnet152()
        self.resnet = self.resnet.to(self.device).eval()

        cfg = 'news_caption_analytic/yolov3/cfg/yolov3-spp.cfg'
        weight_path = 'data/yolov3-spp-ultralytics.pt'
        self.darknet = Darknet(cfg, img_size=416)
        attempt_download(weight_path)
        self.darknet.load_state_dict(torch.load(
            weight_path, map_location=self.device)['model'])
        self.darknet.to(self.device).eval()

        # Get names and colors
        print("get names and colors")
        self.names = load_classes('news_caption_analytic/yolov3/data/coco.names')
        random.seed(123)
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]

        self.preprocess = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        data_iterator = BasicIterator(batch_size=4)
        data_iterator.index_with(model.vocab)
        self.data_iterator = data_iterator

        self.tokenizer = Tokenizer.from_params(
            config.get('dataset_reader').get('tokenizer'))

        indexer_params = config.get('dataset_reader').get('token_indexers')

        self.token_indexers = {}
        for k, p in indexer_params.items():
        	class_type = p.pop("type")
        	print(class_type)
        	self.token_indexers[k] = RobertaTokenIndexer.from_params(p)
    def generate_caption(self, articles):
        """
        article is the input source of what we define
        """
        instances = [self.prepare_instance(a[0], a[1]) for a in articles]
        iterator = self.data_iterator(instances, num_epochs=1, shuffle=False)
        generated_captions = []
        for batch in iterator:
            if self.device.type == 'cuda':
                batch = move_to_device(batch, self.device.index)
            attns_list = self.model.generate(**batch)
            # generated_captions += output_dict['generations']
            # attns = output_dict['attns']
            # len(attns) == gen_len (ignoring seed)
            # len(attns[0]) == n_layers
            # attns[0][0]['image'].shape == [47]
            # attns[0][0]['article'].shape == [article_len]

        output = []
        for i, instance in enumerate(instances):
            buffered = BytesIO()
            instance['metadata']['image'].save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            output.append({
                'title': instance['metadata']['title'],
                'start': instance['metadata']['start'],
                'before': instance['metadata']['before'],
                'after': instance['metadata']['after'],
                # 'caption': generated_captions[i],
                'attns': attns_list[i],
                'image': img_str,
            })

        return output
    def prepare_instance(self, article, article_id=None):
        #This is really important
        sample = self.prepare_sample(article, article_id)
        context = '\n'.join(sample['paragraphs']).strip()
        context_tokens = self.tokenizer.tokenize(context)

        fields = {
            # 'context': CopyTextField(context_tokens, self.token_indexers, proper_infos, proper_infos, 'context'),
            'context': TextField(context_tokens, self.token_indexers),
            'image': ImageField(sample['image'], self.preprocess),
            'face_embeds': ArrayField(sample['face_embeds'], padding_value=np.nan),
            'obj_embeds': ArrayField(sample['obj_embeds'], padding_value=np.nan),
        }

        metadata = {
            'title': sample['title'],
            'start': '\n'.join(sample['start']).strip(),
            'before': '\n'.join(sample['before']).strip(),
            'after': '\n'.join(sample['after']).strip(),
            'image': CenterCrop(224)(Resize(256)(sample['image']))
        }
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def to_token_ids(self, sentence):
        bpe_tokens = self.bpe.encode(sentence)
        words = tokenize_line(bpe_tokens)

        token_ids = []
        for word in words:
            idx = self.indices[word]
            token_ids.append(idx)
        return token_ids
    def get_faces(self, image):
        with torch.no_grad():
            try:
                faces = self.mtcnn(image)
            except IndexError:  # Strange index error on line 135 in utils/detect_face.py
                logger.warning('Strange index error from FaceNet.')
                return np.array([[]])

            if faces is None:
                return np.array([[]])

            embeddings, _ = self.inception(faces)
            return embeddings.cpu().numpy()[:4]
    def get_objects(self, image):
        im0 = np.array(image)
        img = letterbox(im0, new_shape=416)[0]
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.darknet(img)[0]

        # Apply NMS
        # We ignore the person class (class 0)
        pred = non_max_suppression(pred, 0.3, 0.6,
                                   classes=None, agnostic=False)

        # Process detections
        assert len(pred) == 1, f'Length of pred is {len(pred)}'
        det = pred[0]

        im0 = im0[:, :, ::-1]  # to BGR
        im0 = np.ascontiguousarray(im0)

        obj_feats = []
        confidences = []
        classes = []
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for j, (*xyxy, conf, class_) in enumerate(det):
                if j >= 64:
                    break

                obj_feat = get_obj_embeddings(
                    xyxy, image, None, self.resnet)
                obj_feats.append(obj_feat)
                confidences.append(conf.item())
                classes.append(int(class_))

                label = '%s %.2f' % (self.names[int(class_)], conf)
                plot_one_box(xyxy, im0, label=label,
                             color=self.colors[int(class_)])

        # Save results (image with detections)
        # cv2.imwrite(save_path, im0)

        if not obj_feats:
            return np.array([[]])
        return np.array(obj_feats)
    def prepare_sample(self, article, article_id=None):
        assert article_id is not None, "article id cannot be None"
        paragraphs = []
        start = []
        n_words = 0
        
        #Get the location of the first figure
        sections = article['content']
        article_figures = article['figures']
        figure_map = {x['media'][0]['source_uri']:x['media'][0]['uri'] for x in article_figures}
        #Get the position of the figures
        for i, content_item in enumerate(sections):
            if content_item['Type'] == "Figure":
                pos = i
                assert content_item['Media']
                assert content_item['Media'][0]['Type'] == "image"
                source_uri = content_item['Media'][0]['SourceUri']
                image_path = figure_map[source_uri]
                break

        #Add title into the data
        if article['title']:
            paragraphs.append(article['title'])
            n_words += len(self.to_token_ids(article['title']))
            
        #Define before and after
        before = []
        after = []
        i = pos - 1
        j = pos + 1
        
        # Append the first paragraph
        for k, section in enumerate(sections):
            if section['Type'] == 'Paragraph':
                #concatenate the text together
                current_paragraph_text_list = [y["Content"][0].strip() for y in section["Content"] if y['Type'] == "Text"]
                current_paragraph_text = ''.join(current_paragraph_text_list)
                paragraphs.append(current_paragraph_text)
                start.append(current_paragraph_text)
                n_words += len(self.to_token_ids(current_paragraph_text))
                break
                
        #Add the paragraphs around the image
        while True:
            if i > k and sections[i]['Type'] == 'Paragraph':
                #text = sections[i]['text']
                text_list = [y["Content"][0].strip() for y in sections[i]["Content"] if y['Type'] == "Text"]
                text = ''.join(text_list)
                before.insert(0, text)
                n_words += len(self.to_token_ids(text))
            i -= 1

            if k < j < len(sections) and sections[j]['Type'] == 'Paragraph':
                #text = sections[j]['text']
                text_list = [y["Content"][0].strip() for y in sections[j]["Content"] if y['Type'] == "Text"]
                text = ''.join(text_list)
                after.append(text)
                n_words += len(self.to_token_ids(text))
            j += 1

            if n_words >= 510 or (i <= k and j >= len(sections)):
                break
        
        #Prepare the Image
        full_image_path = '/'.join([self.news_data_path, article_id, image_path])
        with Image.open(full_image_path) as image:
            image = image.convert('RGB')
        #get the face embeds
        face_embeds = self.get_faces(image)
        #get the object embeds]
        obj_embeds = self.get_objects(image)
        
        output = {
            'paragraphs': paragraphs + before + after,
            'title': article['title'],
            'start': start,
            'before': before,
            'after': after,
            'image': image,
            'face_embeds': face_embeds,
            'obj_embeds': obj_embeds,
        }

        return output
        
A = Captioner()
A.initialize()

data_dir = "data/semafor/72de494ed44b2515709b19702a4077fd14050d90005284887800474a2e8d1838"
full_data_path = os.path.join(data_dir, "72de494ed44b2515709b19702a4077fd14050d90005284887800474a2e8d1838.json")
sample_data = json.load(open(full_data_path, "r"))
sample_data_id = "72de494ed44b2515709b19702a4077fd14050d90005284887800474a2e8d1838"
sample_articles = [(sample_data, sample_data_id)]
output_dict = A.generate_caption(sample_articles)
print(output_dict)
#output = A.prepare_sample(sample_data, article_id="72de494ed44b2515709b19702a4077fd14050d90005284887800474a2e8d1838")