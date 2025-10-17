"""
Système de détection d'EPI (Équipements de Protection Individuelle)
avec YOLO et description en langage naturel via LLM
"""

import cv2
import numpy as np
import requests
import base64
from pathlib import Path
from typing import List, Dict, Tuple
import json
import argparse
import os
from datetime import datetime

class EPIDetectionSystem:
    def __init__(self, ollama_url: str = "http://localhost:11434", 
                 model_name: str = "llama3.2-vision:latest",
                 yolo_model_path: str = None):
        """
        Initialise le système de détection d'EPI
        
        Args:
            ollama_url: URL du serveur Ollama
            model_name: Nom du modèle Ollama (llama3.2-vision, llava, etc.)
            yolo_model_path: Chemin vers le modèle YOLO personnalisé (optionnel)
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Vérifier la connexion à Ollama
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                print(f"✓ Connecté à Ollama sur {ollama_url}")
                models = response.json().get('models', [])
                print(f"  Modèles disponibles: {[m['name'] for m in models]}")
            else:
                print(f"⚠ Ollama accessible mais réponse inattendue")
        except requests.exceptions.ConnectionError:
            print(f"⚠ Impossible de se connecter à Ollama sur {ollama_url}")
            print("  Assurez-vous qu'Ollama est lancé: ollama serve")
        
        # Classes d'EPI à détecter (restreintes)
        self.epi_classes = [
            'casque', 'gilet_haute_visibilite', 'lunettes_protection',
            'gants', 'bouchons_oreilles', 'chaussures_securite',
            'masque', 'visiere_protection', 'personne'
        ]

        # mapping réduit des labels habituels vers les classes EPI pertinentes
        self.label_to_epi = {
            # casque
            'helmet': 'casque', 'hard hat': 'casque', 'hardhat': 'casque', 'safety helmet': 'casque',
            # gilet haute visibilité
            'safety vest': 'gilet_haute_visibilite', 'vest': 'gilet_haute_visibilite', 'high visibility vest': 'gilet_haute_visibilite', 'yellow vest': 'gilet_haute_visibilite', 'reflective vest': 'gilet_haute_visibilite',
            # gants
            'glove': 'gants', 'gloves': 'gants', 'safety glove': 'gants', 'work glove': 'gants',
            # lunettes / protections oculaires
            'safety glasses': 'lunettes_protection', 'goggles': 'lunettes_protection', 'sunglasses': 'lunettes_protection',
            # bouchons / protections auditives
            'earplug': 'bouchons_oreilles', 'ear plugs': 'bouchons_oreilles', 'ear protector': 'bouchons_oreilles',
            # bottes / chaussures de sécurité
            'boot': 'chaussures_securite', 'boots': 'chaussures_securite', 'shoe': 'chaussures_securite', 'shoes': 'chaussures_securite', 'bottes': 'chaussures_securite',
            # masque / respirateur
            'mask': 'masque', 'face mask': 'masque', 'respirator': 'masque', 'masque': 'masque',
            # visière / face shield
            'visor': 'visiere_protection', 'face shield': 'visiere_protection', 'visiere': 'visiere_protection', 'visière': 'visiere_protection', 'face-shield': 'visiere_protection',
            # personne (utile pour heuristiques)
            'person': 'personne', 'personne': 'personne'
        }
        
        # Initialisation de YOLO (YOLOv8 via ultralytics)
        try:
            from ultralytics import YOLO
            if yolo_model_path:
                self.model = YOLO(yolo_model_path)
            else:
                self.model = YOLO('yolov8n.pt')
            print("✓ Modèle YOLO chargé avec succès")
        except Exception as e:
            # message concis si ultralytics absent ou échec de chargement
            print("⚠ ultralytics non disponible ou échec de chargement du modèle:", str(e))
            self.model = None
    
    def map_label_to_epi(self, label: str) -> str:
        """
        Normalise un label retourné par YOLO vers une classe EPI canonique si possible.
        Retourne le label mappé ou le label original (minuscules) si non mappé.
        """
        if not label:
            return label
        key = label.lower().strip()
        # suppression ponctuation basique
        key = ''.join(ch for ch in key if ch.isalnum() or ch.isspace())
        mapped = self.label_to_epi.get(key)
        return mapped if mapped else key

    def _detect_high_vis_on_person(self, image: np.ndarray, bbox: List[int], yellow_thresh: float = 0.12) -> Tuple[bool, float]:
        """
        Heuristique : coupe la zone torse et mesure la proportion de pixels jaunes (HSV) pour estimer un gilet.
        Retourne (détecté_bool, proportion).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        # clamp
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
        # définir zone torse approximative: zone centrale verticale (30%-80% hauteur du bbox)
        height = y2 - y1
        torso_y1 = int(y1 + 0.3 * height)
        torso_y2 = int(y1 + 0.8 * height)
        torso_x1 = x1
        torso_x2 = x2
        crop = image[torso_y1:torso_y2, torso_x1:torso_x2]
        if crop.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # plages pour jaune (ajustable)
        lower_y = np.array([15, 80, 80])
        upper_y = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_y, upper_y)
        ratio = float(np.count_nonzero(mask)) / (mask.size + 1e-8)
        return (ratio >= yellow_thresh), ratio

    def _detect_helmet_on_person(self, image: np.ndarray, bbox: List[int], thresh: float = 0.06) -> Tuple[bool, float]:
        """
        Heuristique casque : recherche de pixels jaunes/oranges/blancs/bleus dans la zone tête (top 0-30% du bbox).
        Retourne (détecté_bool, ratio).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.0
        head_h = max(2, int(0.30 * (y2 - y1)))
        crop = image[y1:y1+head_h, x1:x2]
        if crop.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # masks: yellow, orange, blue, white
        lower_y, upper_y = np.array([15, 80, 80]), np.array([40, 255, 255])
        lower_o, upper_o = np.array([5, 80, 80]), np.array([15, 255, 255])
        lower_b, upper_b = np.array([90, 50, 50]), np.array([130, 255, 255])
        # white: low sat, high value
        white_mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,50,255]))
        mask = cv2.inRange(hsv, lower_y, upper_y) | cv2.inRange(hsv, lower_o, upper_o) | cv2.inRange(hsv, lower_b, upper_b) | white_mask
        ratio = float(np.count_nonzero(mask)) / (mask.size + 1e-8)
        return (ratio >= thresh), ratio

    def _detect_gloves_on_person(self, image: np.ndarray, bbox: List[int], side_thresh: float = 0.03) -> Tuple[bool, float, List[int]]:
        """
        Heuristique gants : vérifie zones latérales basses pour colorations typiques (jaune/orange/blanc/bleu).
        Retourne (détecté_bool, ratio_max, bbox_det) où bbox_det est bbox approximative de la main détectée.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.0, []
        width = x2 - x1
        height = y2 - y1
        # zones mains approximatives : gauche et droite, vertical 45%-80%
        vy1 = int(y1 + 0.45*height); vy2 = int(y1 + 0.80*height)
        left_x1 = x1; left_x2 = x1 + int(0.22*width)
        right_x1 = x2 - int(0.22*width); right_x2 = x2
        crops = [
            (image[vy1:vy2, left_x1:left_x2], (left_x1,vy1,left_x2,vy2)),
            (image[vy1:vy2, right_x1:right_x2], (right_x1,vy1,right_x2,vy2))
        ]
        best_ratio = 0.0; best_box = []
        for crop, box in crops:
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower_y, upper_y = np.array([15, 80, 80]), np.array([40, 255, 255])
            lower_o, upper_o = np.array([5, 80, 80]), np.array([15, 255, 255])
            lower_b, upper_b = np.array([90, 50, 50]), np.array([130, 255, 255])
            white_mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,50,255]))
            mask = cv2.inRange(hsv, lower_y, upper_y) | cv2.inRange(hsv, lower_o, upper_o) | cv2.inRange(hsv, lower_b, upper_b) | white_mask
            ratio = float(np.count_nonzero(mask)) / (mask.size + 1e-8)
            if ratio > best_ratio:
                best_ratio = ratio
                best_box = box
        return (best_ratio >= side_thresh), best_ratio, best_box

    def detect_epi(self, image_path: str, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
        """
        Détecte les EPI sur une image. Accepté: chemin ou numpy array (si besoin).
        Retourne image annotée (BGR numpy array) et liste de détections.
        """
        if self.model is None:
            raise RuntimeError("Modèle YOLO non initialisé")
        
        # Chargement de l'image (si chemin fourni)
        image = None
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
        elif isinstance(image_path, np.ndarray):
            image = image_path.copy()
        else:
            raise ValueError("image_path doit être un chemin ou un numpy.ndarray")
        
        # Exécution du modèle YOLO
        results = self.model(image, conf=conf_threshold)
        
        # Extraire détections d'une manière robuste (gère différentes versions ultralytics)
        detections: List[Dict] = []
        allowed_set = set(self.epi_classes)  # filtre strict
        
        try:
            # results peut être itérable ; on prend le premier résultat si nécessaire
            first = results[0] if len(results) > 0 else results
            # extraire arrays si disponibles
            boxes_xyxy = getattr(first.boxes, 'xyxy', None)
            if boxes_xyxy is not None:
                xyxy_array = boxes_xyxy.cpu().numpy() if hasattr(boxes_xyxy, 'cpu') else np.array(boxes_xyxy)
                confs = getattr(first.boxes, 'conf', None)
                cls_probs = getattr(first.boxes, 'cls', None)
                confs_array = confs.cpu().numpy() if confs is not None and hasattr(confs, 'cpu') else (np.array(confs) if confs is not None else np.ones((xyxy_array.shape[0],)))
                cls_array = cls_probs.cpu().numpy() if cls_probs is not None and hasattr(cls_probs, 'cpu') else (np.array(cls_probs) if cls_probs is not None else np.zeros((xyxy_array.shape[0],)))
                names = getattr(first, 'names', {})
                # essayer de récupérer des masques (pour affiner les bbox) - format variable selon ultralytics
                masks_data = None
                masks_attr = getattr(first, 'masks', None)
                if masks_attr is not None:
                    # plusieurs formes possibles : masks.data (tensor), masks (list/np)
                    try:
                        maybe = getattr(masks_attr, 'data', masks_attr)
                        # tenter cpu->numpy si tensor
                        if hasattr(maybe, 'cpu'):
                            masks_data = maybe.cpu().numpy()
                        else:
                            masks_data = np.array(maybe)
                    except Exception:
                        masks_data = None
                for i, box in enumerate(xyxy_array):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    conf = float(confs_array[i])
                    cls = int(cls_array[i])
                    class_name = names.get(cls, str(cls))
                    if conf < conf_threshold:
                        continue
                    # raffinement: si masque disponible, calculer bbox serrée depuis le masque i
                    if masks_data is not None and i < len(masks_data):
                        try:
                            mask_i = masks_data[i]
                            # mask peut être bool/0-1 ou float ; s'assurer binaire
                            mask_bin = (mask_i > 0.5).astype(np.uint8)
                            ys, xs = np.where(mask_bin)
                            if ys.size > 0 and xs.size > 0:
                                x1m, x2m = int(xs.min()), int(xs.max())
                                y1m, y2m = int(ys.min()), int(ys.max())
                                # clamp to image
                                h_img, w_img = image.shape[:2]
                                x1, y1 = max(0, x1m), max(0, y1m)
                                x2, y2 = min(w_img-1, x2m), min(h_img-1, y2m)
                        except Exception:
                            # fallback : conserver bbox initiale
                            pass
                    mapped = self.map_label_to_epi(class_name)
                    # n'ajouter QUE si mappé vers une classe autorisée (ou 'personne')
                    if mapped in allowed_set:
                        detections.append({
                            'class': class_name,
                            'epi_class': mapped,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
            else:
                # Fallback : tenter d'extraire à partir des attributs des boxes un par un
                for res in results:
                    if not hasattr(res, 'boxes'):
                        continue
                    for b in res.boxes:
                        try:
                            xy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy, 'cpu') else np.array(b.xyxy[0])
                            conf = float(b.conf[0]) if hasattr(b.conf) else float(b.conf)
                            cls = int(b.cls[0]) if hasattr(b.cls) else int(b.cls)
                            class_name = res.names.get(cls, str(cls))
                            if conf < conf_threshold:
                                continue
                            x1, y1, x2, y2 = map(int, xy.tolist())
                            mapped = self.map_label_to_epi(class_name)
                            if mapped in allowed_set:
                                detections.append({
                                    'class': class_name,
                                    'epi_class': mapped,
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2]
                                })
                        except Exception:
                            continue
        except Exception as e:
            # si extraction échoue, on renvoie une liste vide mais sans planter le pipeline
            print("⚠ Extraction des détections YOLO échouée:", str(e))
        
        # heuristique: si personne détectée et aucun gilet nearby -> vérifier couleur jaune sur torse
        # on parcourt les personnes et on ajoute détection gilet si couleur trouvée
        persons = [d for d in detections if d.get('epi_class') == 'personne' or d.get('class','').lower() == 'person']
        existing_vests_boxes = [d['bbox'] for d in detections if d.get('epi_class') == 'gilet_haute_visibilite']
        # also collect existing helmets, gloves to avoid duplicates
        existing_helmets = [d['bbox'] for d in detections if d.get('epi_class') == 'casque']
        existing_gloves = [d['bbox'] for d in detections if d.get('epi_class') == 'gants']

        for p in persons:
            # skip si un gilet existe déjà qui chevauche le bbox de la personne (simple IoU-like)
            px1, py1, px2, py2 = p['bbox']
            has_vest = False
            for vb in existing_vests_boxes:
                vx1, vy1, vx2, vy2 = vb
                ix1, iy1 = max(px1, vx1), max(py1, vy1)
                ix2, iy2 = min(px2, vx2), min(py2, vy2)
                if ix2 > ix1 and iy2 > iy1:
                    has_vest = True
                    break
            if not has_vest:
                detected, ratio = self._detect_high_vis_on_person(image, p['bbox'])
                if detected:
                    vx1 = int(p['bbox'][0] + 0.1*(p['bbox'][2]-p['bbox'][0]))
                    vy1 = int(p['bbox'][1] + 0.3*(p['bbox'][3]-p['bbox'][1]))
                    vx2 = int(p['bbox'][2] - 0.1*(p['bbox'][2]-p['bbox'][0]))
                    vy2 = int(p['bbox'][1] + 0.8*(p['bbox'][3]-p['bbox'][1]))
                    detections.append({
                        'class': 'color_yellow_patch',
                        'epi_class': 'gilet_haute_visibilite',
                        'confidence': float(min(0.99, ratio*2.0)),
                        'bbox': [vx1, vy1, vx2, vy2]
                    })
            # casque
            has_helmet = False
            for hb in existing_helmets:
                hx1, hy1, hx2, hy2 = hb
                ix1, iy1 = max(px1, hx1), max(py1, hy1)
                ix2, iy2 = min(px2, hx2), min(py2, hy2)
                if ix2 > ix1 and iy2 > iy1:
                    has_helmet = True
                    break
            if not has_helmet:
                detected_h, ratio_h = self._detect_helmet_on_person(image, p['bbox'])
                if detected_h:
                    # create small head bbox
                    hx1 = px1; hy1 = py1; hx2 = px2; hy2 = py1 + int(0.30*(py2-py1))
                    detections.append({
                        'class': 'color_head_patch',
                        'epi_class': 'casque',
                        'confidence': float(min(0.99, ratio_h*2.0)),
                        'bbox': [hx1, hy1, hx2, hy2]
                    })
            # gants
            has_glove = False
            for gb in existing_gloves:
                gx1, gy1, gx2, gy2 = gb
                ix1, iy1 = max(px1, gx1), max(py1, gy1)
                ix2, iy2 = min(px2, gx2), min(py2, gy2)
                if ix2 > ix1 and iy2 > iy1:
                    has_glove = True
                    break
            if not has_glove:
                detected_g, ratio_g, gbox = self._detect_gloves_on_person(image, p['bbox'])
                if detected_g:
                    # if bbox available use it else approximate
                    gbbox = gbox if gbox else [px1, int(py1+0.6*(py2-py1)), px1+int(0.15*(px2-px1)), int(py1+0.8*(py2-py1))]
                    detections.append({
                        'class': 'color_hand_patch',
                        'epi_class': 'gants',
                        'confidence': float(min(0.99, ratio_g*2.0)),
                        'bbox': gbbox
                    })

        # Annotation de l'image avec OpenCV (inchangé sauf lecture de 'epi_class' et couleurs adaptées)
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det.get('epi_class') or det.get('class')
            conf = det['confidence']
            # couleur verte pour les EPI autorisés, orange autrement (ne devrait pas arriver)
            color = (0, 200, 0) if cls_name in {'casque','gilet_haute_visibilite','gants','lunettes_protection','Casque_anti_bruit','chaussures_securite','masque','visiere_protection'} else (0, 120, 255)
            # rectangle plus précis (coords int)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, lineType=cv2.LINE_AA)
            # label plus discret : taille réduite et petit padding
            label = f"{cls_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # plus petit
            font_thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            pad = 4
            lx1 = int(x1)
            ly1 = int(y1) - th - pad
            ly2 = int(y1)
            lx2 = int(x1) + tw + pad
            if ly1 < 0:
                # si trop haut, dessiner sous la bbox
                ly1 = int(y2)
                ly2 = int(y2 + th + pad)
            cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, cv2.FILLED, lineType=cv2.LINE_AA)
            text_x = lx1 + 2
            text_y = ly2 - 3 if ly2 > ly1 else ly2
            cv2.putText(annotated, label, (text_x, text_y), font, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

        return annotated, detections
    
    def analyze_epi_compliance(self, detections: List[Dict]) -> Dict:
        """
        Analyse la conformité des EPI détectés
        
        Args:
            detections: Liste des détections
            
        Returns:
            Analyse de conformité
        """
        epi_detected = {}
        persons_detected = 0
        allowed_epi = set(self.epi_classes) - {'personne'}
        
        for det in detections:
            class_name = det.get('epi_class', det.get('class'))
            # compter les personnes
            if class_name == 'personne' or 'person' in str(class_name).lower():
                persons_detected += 1
                continue
            # n'incrémenter que si dans la liste autorisée
            if class_name in allowed_epi:
                epi_detected[class_name] = epi_detected.get(class_name, 0) + 1
        
        analysis = {
            'personnes_detectees': persons_detected,
            'epi_detectes': epi_detected,
            'total_epi': sum(epi_detected.values()),
            'detections_brutes': detections
        }
        return analysis
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode une image en base64 pour Ollama"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def query_ollama(self, prompt: str, image_path: str = None) -> str:
        """
        Interroge Ollama avec ou sans image
        
        Args:
            prompt: Prompt texte
            image_path: Chemin vers l'image (optionnel, pour modèles vision)
            
        Returns:
            Réponse du modèle
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if image_path:
            base64_image = self.encode_image_to_base64(image_path)
            payload["images"] = [base64_image]
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                resp_json = response.json()
                # fallbacks pour différents formats de réponse
                if 'response' in resp_json:
                    return resp_json['response']
                if 'choices' in resp_json and isinstance(resp_json['choices'], list) and len(resp_json['choices'])>0:
                    # Ollama-like or OpenAI-like fallback
                    c = resp_json['choices'][0]
                    return c.get('text') or c.get('message') or json.dumps(c, ensure_ascii=False)
                # last resort: raw text
                return resp_json.get('output') or json.dumps(resp_json, ensure_ascii=False)
            else:
                return f"Erreur Ollama: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Erreur lors de la requête Ollama: {str(e)}"
    
    def generate_natural_description(self, 
                                     image_path: str, 
                                     detections: List[Dict],
                                     analysis: Dict,
                                     use_vision: bool = True) -> str:
        """
        Génère une description en langage naturel avec Ollama (modifié pour rapport synthétique)
        """
        # Préparation du contexte (JSON compact)
        detection_summary = json.dumps(analysis, ensure_ascii=False)

        # Prompt strict : une seule phrase en français, rapport synthétique uniquement
        concise_instruction = (
            "Tu es un assistant. FOURNIS UNIQUEMENT UNE PHRASE SYNTHÉTIQUE EN FRANÇAIS "
            "qui RÉSUME le rapport de détection en t'appuyant sur les données fournies. "
            "Construis la phrase à partir des nombres et types d'EPI détectés; n'utilise pas "
            "un exemple littéralement ni de texte additionnel. La phrase doit refléter le contenu "
            "des données (nombre de personnes et EPI par personne si possible)."
        )

        prompt = f"""{concise_instruction}

Données:
{detection_summary}

Réponds PAR UNE SEULE PHRASE EN FRANÇAIS, adaptée aux données ci‑dessus (ne répète pas un exemple littéralement)."""

        if use_vision:
            response = self.query_ollama(prompt, image_path)
        else:
            response = self.query_ollama(prompt)

        # Nettoyage simple de la réponse
        if isinstance(response, str):
            return response.strip()
        # fallback minimal si format inattendu
        try:
            return str(response).strip()
        except Exception:
            return "Erreur: réponse LLM inattendue."
    
    def process_image(self, 
                     image_path: str, 
                     output_path: str = None,
                     json_output: str = None,
                     conf_threshold: float = 0.5,
                     use_vision: bool = True) -> Dict:
        """
        Pipeline complet : détection + analyse + description
        
        Args:
            image_path: Chemin vers l'image d'entrée
            output_path: Chemin pour sauvegarder l'image annotée
            json_output: Chemin pour sauvegarder les résultats au format JSON
            conf_threshold: Seuil de confiance pour la détection
            use_vision: Utiliser GPT-4 Vision
            
        Returns:
            Résultats complets
        """
        print(f"🔍 Analyse de l'image: {image_path}")
        image_annotee, detections = self.detect_epi(image_path, conf_threshold=conf_threshold)
        print("✅ Analyse de conformité...")
        analysis = self.analyze_epi_compliance(detections)
        print("💬 Génération de la description avec LLM...")
        description = self.generate_natural_description(
            image_path, detections, analysis, use_vision
        )
        if output_path:
            cv2.imwrite(output_path, image_annotee)
            print(f"💾 Image annotée sauvegardée: {output_path}")
        # sauvegarde JSON si demandé
        result_package = {
            'detections': detections,
            'analyse': analysis,
            'description_naturelle': description,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        if json_output:
            try:
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(result_package, f, ensure_ascii=False, indent=2)
                print(f"💾 Résultats JSON sauvegardés: {json_output}")
            except Exception as e:
                print("⚠ Échec sauvegarde JSON:", str(e))
        return {
            'image_annotee': image_annotee,
            'detections': detections,
            'analyse': analysis,
            'description_naturelle': description
        }

# ============================================================================ 
# Exemple d'utilisation CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Détection d'EPI + description LLM")
    parser.add_argument("image", help="Chemin vers l'image à analyser")
    parser.add_argument("--yolo", help="Chemin du modèle YOLO (.pt)", default=None)
    parser.add_argument("--out", help="Image annotée de sortie", default="resultat_detection.jpg")
    parser.add_argument("--json", help="Fichier JSON de sortie", default="resultat_detection.json")
    parser.add_argument("--conf", help="Seuil de confiance (0-1)", type=float, default=0.5)
    parser.add_argument("--no-vision", help="Ne pas envoyer l'image au LLM (texte uniquement)", action="store_true")
    args = parser.parse_args()

    system = EPIDetectionSystem(
        ollama_url="http://localhost:11434",
        model_name="llama3.2-vision:latest",
        yolo_model_path=args.yolo
    )

    try:
        results = system.process_image(
            image_path=args.image,
            output_path=args.out,
            json_output=args.json,
            conf_threshold=args.conf,
            use_vision=not args.no_vision
        )
        # affichage synthétique
        print("\nRAPPORT:")
        print(f" Personnes détectées: {results['analyse']['personnes_detectees']}")
        print(f" Total EPI détectés: {results['analyse']['total_epi']}")
        print(" EPI par type:")
        for k,v in results['analyse']['epi_detectes'].items():
            print(f"  - {k}: {v}")
        print("\nDescription LLM:")
        print(results['description_naturelle'])
    except Exception as e:
        print("❌ Erreur:", str(e))


if __name__ == "__main__":
    # Instructions d'installation
    print("""
    📦 Packages requis:
    pip install ultralytics opencv-python pillow requests
    
    🔧 Installation d'Ollama:
    1. Télécharger Ollama: https://ollama.ai/download
    2. Lancer Ollama: ollama serve
    3. Télécharger un modèle vision:
       ollama pull llama3.2-vision:latest
       # ou
       ollama pull llava:latest
       # ou pour texte seul (plus léger):
       ollama pull llama3.2:latest
    
    📝 Modèles Ollama recommandés:
    - llama3.2-vision:latest (11B) - Excellent pour l'analyse d'image, français OK
    - llava:latest (7B) - Bon compromis, multilingue
    - llama3.2:latest (3B) - Léger, texte seul, bon français
    - llama3.1:8b - Très bon en français, texte seul
    
    💡 Comparaison des modèles vision:
    - llama3.2-vision: Meilleure compréhension, plus lent (~11GB)
    - llava: Plus rapide, bon compromis (~4.5GB)
    
    📝 Notes importantes:
    1. Le modèle YOLOv8 de base détecte des personnes mais pas les EPI spécifiques
    2. Pour une détection précise des EPI, il faut fine-tuner YOLO sur un dataset d'EPI
    3. Ollama tourne 100% en local, aucun coût ni limite d'API !
    
    🎯 Pour améliorer les performances:
    - Fine-tuner YOLOv8 sur un dataset d'EPI annoté
    - Utiliser llama3.2-vision pour une analyse d'image plus précise
    - Ajuster les prompts selon vos besoins spécifiques
    """)
    
    # Décommenter pour lancer
    main()