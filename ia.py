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
        
        # Classes d'EPI à détecter
        self.epi_classes = [
            'casque', 'gilet_haute_visibilite', 'lunettes_protection',
            'gants', 'chaussures_securite', 'masque', 'harnais',
            'bouchons_oreilles', 'personne'
        ]
        
        # Initialisation de YOLO (YOLOv8 via ultralytics)
        try:
            from ultralytics import YOLO
            if yolo_model_path:
                self.model = YOLO(yolo_model_path)
            else:
                # Modèle YOLOv8 pré-entraîné (à fine-tuner pour les EPI)
                self.model = YOLO('yolov8n.pt')
            print("✓ Modèle YOLO chargé avec succès")
        except ImportError:
            print("⚠ ultralytics non installé. Installez avec: pip install ultralytics")
            self.model = None
    
    def detect_epi(self, image_path: str, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
        """
        Détecte les EPI sur une image
        
        Args:
            image_path: Chemin vers l'image
            conf_threshold: Seuil de confiance minimum
            
        Returns:
            image_annotee: Image avec les détections annotées
            detections: Liste des détections avec leurs informations
        """
        if self.model is None:
            raise RuntimeError("Modèle YOLO non initialisé")
        
        # Chargement de l'image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Détection avec YOLO
        results = self.model(image, conf=conf_threshold)
        
        # Extraction des détections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Annotation de l'image
        image_annotee = results[0].plot()
        
        return image_annotee, detections
    
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
        
        for det in detections:
            class_name = det['class']
            if class_name == 'personne' or 'person' in class_name.lower():
                persons_detected += 1
            else:
                if class_name not in epi_detected:
                    epi_detected[class_name] = 0
                epi_detected[class_name] += 1
        
        # Analyse basique
        analysis = {
            'personnes_detectees': persons_detected,
            'epi_detectes': epi_detected,
            'total_epi': len([d for d in detections if d['class'] != 'personne']),
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
        
        # Ajout de l'image si fournie (pour modèles vision comme llava ou llama3.2-vision)
        if image_path:
            base64_image = self.encode_image_to_base64(image_path)
            payload["images"] = [base64_image]
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                return response.json()['response']
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
        Génère une description en langage naturel avec Ollama
        
        Args:
            image_path: Chemin vers l'image
            detections: Détections YOLO
            analysis: Analyse de conformité
            use_vision: Utiliser un modèle vision (llava, llama3.2-vision)
            
        Returns:
            Description en langage naturel
        """
        # Préparation du contexte
        detection_summary = json.dumps(analysis, indent=2, ensure_ascii=False)
        
        if use_vision:
            # Utilisation d'un modèle vision local
            prompt = f"""Tu es un expert en sécurité au travail spécialisé dans l'analyse d'EPI (Équipements de Protection Individuelle).

Analyse cette image de chantier/lieu de travail et les détections automatiques suivantes :

{detection_summary}

Fournis une description professionnelle en français incluant :
1. Le contexte général de la scène
2. Les personnes présentes et leur équipement
3. Les EPI portés (casque, gilet, gants, lunettes, etc.)
4. Les EPI manquants ou non conformes
5. Une évaluation globale de la conformité sécurité
6. Des recommandations si nécessaire

Sois précis, concis et professionnel. Réponds UNIQUEMENT en français."""
            
            response = self.query_ollama(prompt, image_path)
        
        else:
            # Utilisation sans vision (basé uniquement sur les détections)
            prompt = f"""Tu es un expert en sécurité au travail.

Analyse les détections suivantes d'EPI (Équipements de Protection Individuelle) :

{detection_summary}

Fournis un rapport professionnel en français incluant :
1. Résumé des personnes et EPI détectés
2. Évaluation de la conformité sécurité
3. Points positifs
4. Points d'amélioration ou EPI manquants
5. Recommandations

Sois précis et professionnel. Réponds UNIQUEMENT en français."""
            
            response = self.query_ollama(prompt)
        
        return response
    
    def process_image(self, 
                     image_path: str, 
                     output_path: str = None,
                     use_vision: bool = True) -> Dict:
        """
        Pipeline complet : détection + analyse + description
        
        Args:
            image_path: Chemin vers l'image d'entrée
            output_path: Chemin pour sauvegarder l'image annotée
            use_vision: Utiliser GPT-4 Vision
            
        Returns:
            Résultats complets
        """
        print(f"🔍 Analyse de l'image: {image_path}")
        
        # 1. Détection avec YOLO
        print("📊 Détection des EPI avec YOLO...")
        image_annotee, detections = self.detect_epi(image_path)
        
        # 2. Analyse de conformité
        print("✅ Analyse de conformité...")
        analysis = self.analyze_epi_compliance(detections)
        
        # 3. Génération de la description en langage naturel
        print("💬 Génération de la description avec GPT...")
        description = self.generate_natural_description(
            image_path, detections, analysis, use_vision
        )
        
        # 4. Sauvegarde de l'image annotée
        if output_path:
            cv2.imwrite(output_path, image_annotee)
            print(f"💾 Image annotée sauvegardée: {output_path}")
        
        return {
            'image_annotee': image_annotee,
            'detections': detections,
            'analyse': analysis,
            'description_naturelle': description
        }


# ============================================================================
# Exemple d'utilisation
# ============================================================================

def main():
    """Exemple d'utilisation du système"""
    
    # Configuration
    IMAGE_PATH = "images1.jpg"  # Votre image de test
    OUTPUT_PATH = "resultat_detection.jpg"
    
    # Initialisation du système avec Ollama
    system = EPIDetectionSystem(
        ollama_url="http://localhost:11434",
        model_name="llama3.2-vision:latest"  # ou "llava:latest", "llama3.2:latest"
    )
    
    # Traitement de l'image
    try:
        results = system.process_image(
            image_path=IMAGE_PATH,
            output_path=OUTPUT_PATH,
            use_vision=True  # True pour modèle vision, False pour texte seul
        )
        
        # Affichage des résultats
        print("\n" + "="*70)
        print("📋 RAPPORT D'ANALYSE EPI")
        print("="*70)
        print(f"\n🔢 Statistiques:")
        print(f"   • Personnes détectées: {results['analyse']['personnes_detectees']}")
        print(f"   • Total EPI détectés: {results['analyse']['total_epi']}")
        print(f"\n🎯 EPI identifiés:")
        for epi, count in results['analyse']['epi_detectes'].items():
            print(f"   • {epi}: {count}")
        
        print(f"\n📝 DESCRIPTION DÉTAILLÉE:")
        print("-" * 70)
        print(results['description_naturelle'])
        print("="*70)
        
        # Affichage de l'image (optionnel)
        # cv2.imshow("Detection EPI", results['image_annotee'])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


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