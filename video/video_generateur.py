#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
import re
from moviepy.editor import ImageSequenceClip, AudioFileClip, TextClip, CompositeVideoClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from gtts import gTTS
import PIL
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- Configuration ---
DEFAULT_TEMP_DIR = "video/temp"
DEFAULT_OUTPUT_FILE = "video_generee.mp4"
DEFAULT_FPS = 24

def clean_directory(dir_path):
    """Supprime et recrée un répertoire."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print(f"🧹 Répertoire temporaire nettoyé et prêt : {dir_path}")

def resize_image(image_path, size, temp_dir):
    """Redimensionne une image à la taille spécifiée et la sauvegarde dans le répertoire temporaire."""
    try:
        img = Image.open(image_path)
        # Ajoute un fond noir si l'image a de la transparence (RGBA)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGBA')
            bg = Image.new('RGB', img.size, (0, 0, 0))
            bg.paste(img, (0, 0), img)
            img = bg
        
        img_resized = img.resize(size, Image.LANCZOS)
        
        temp_image_path = os.path.join(temp_dir, os.path.basename(image_path))
        img_resized.save(temp_image_path)
        return temp_image_path
    except Exception as e:
        print(f"⚠️ Erreur lors du redimensionnement de l'image {image_path}: {e}")
        return None

def generate_audio_from_text(text, audio_path):
    """Génère un fichier audio MP3 à partir d'un texte avec gTTS."""
    print(f"🔊 Génération de l'audio à partir du texte...")
    tts = gTTS(text=text, lang='fr')
    tts.save(audio_path)
    print(f"✅ Audio généré : {audio_path}")

def create_text_image(text, size=(1920, 1080), font_size=70, font_color="white", bg_color=(0,0,0,0)):
    """Crée une image PNG avec le texte centré, utilisable comme overlay."""
    img = Image.new("RGBA", size, bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    # Utiliser textbbox pour mesurer le texte
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size[0]-w)//2, (size[1]-h)//2), text, font=font, fill=font_color)
    temp_path = os.path.join(DEFAULT_TEMP_DIR, "text_overlay.png")
    img.save(temp_path)
    return temp_path

def parse_sections(text_path):
    """Découpe le texte en sections (titre, texte) en ignorant la durée et la ligne image."""
    with open(text_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    sections = []
    i = 0
    while i < len(lines):
        # Cherche un titre de section (ex: 4. Titre (durée))
        m = re.match(r'^(\d+\.\s+)?(.+?)(\s*\(\d+:\d+\s*-\s*\d+:\d+\))?$', lines[i])
        if m and not lines[i].startswith('('):
            titre = m.group(2).strip()
            i += 1
            # Ignore la ligne image si présente
            if i < len(lines) and lines[i].startswith("(Image"):
                i += 1
            # Prend le texte associé
            texte = ""
            while i < len(lines) and not re.match(r'^(\d+\.\s+)?(.+?)(\s*\(\d+:\d+\s*-\s*\d+:\d+\))?$', lines[i]):
                texte += lines[i] + " "
                i += 1
            sections.append({"titre": titre, "texte": texte.strip()})
        else:
            i += 1
    return sections

def enhance_image(img_path, size=(1920,1080)):
    """Améliore l'image (contraste, saturation, flou d'arrière-plan) et la redimensionne."""
    img = Image.open(img_path).convert("RGB").resize(size, Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.2)
    # Flou léger pour le fond
    blurred = img.filter(ImageFilter.GaussianBlur(radius=8))
    # Superpose l'image nette au centre
    img.paste(img.resize((int(size[0]*0.8), int(size[1]*0.8)), Image.LANCZOS), (int(size[0]*0.1), int(size[1]*0.1)))
    return blurred

def overlay_text(img, titre, texte, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
    """Ajoute un titre et un texte sur l'image avec fond semi-transparent pour la lisibilité."""
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    # Fond semi-transparent pour le texte
    overlay_h = int(H * 0.32)
    overlay = Image.new("RGBA", (W, overlay_h), (0,0,0,160))
    img.paste(overlay, (0, H-overlay_h), overlay)
    # Titre
    font_titre = ImageFont.truetype(font_path, 60)
    font_texte = ImageFont.truetype(font_path, 38)
    y = H - overlay_h + 30
    draw.text((60, y), titre, font=font_titre, fill="white", stroke_width=2, stroke_fill="black")
    # Texte multi-ligne
    y += 80
    lines = []
    max_width = W - 120
    words = texte.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        w, _ = draw.textbbox((0,0), test_line, font=font_texte)[2:]
        if w > max_width:
            lines.append(line)
            line = word + " "
        else:
            line = test_line
    lines.append(line)
    for l in lines:
        draw.text((60, y), l, font=font_texte, fill="white", stroke_width=1, stroke_fill="black")
        y += 45
    return img

def make_video_from_sections(sections, images_dir, audio_path, output_path):
    clips = []
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    for i, section in enumerate(sections):
        # Associer une image à chaque section (ou réutiliser si moins d'images)
        img_path = os.path.join(images_dir, image_files[i % len(image_files)])
        img = enhance_image(img_path)
        img = overlay_text(img, section["titre"], section["texte"])
        temp_img_path = f"video/temp/slide_{i}.png"
        img.save(temp_img_path)
        # Animation d'apparition (zoom léger)
        clip = ImageClip(temp_img_path).set_duration(7).resize(width=1920).fx(lambda c: c.crossfadein(1))
        clips.append(clip)
    video = concatenate_videoclips(clips, method="compose")
    # Ajouter l'audio
    if os.path.exists(audio_path):
        audio = AudioFileClip(audio_path)
        video = video.set_audio(audio)
    video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

def generate_audio_from_sections(sections, audio_path):
    """Concatène tous les textes et génère un audio global."""
    full_text = " ".join([s["titre"] + ". " + s["texte"] for s in sections])
    tts = gTTS(text=full_text, lang='fr')
    tts.save(audio_path)

def main(text_file, images_dir, output_file):
    """
    Génère une vidéo à partir d'un texte (fichier), un répertoire d'images et un audio généré automatiquement.
    """
    print("🚀 Démarrage du générateur de vidéo...")

    # --- 1. Lire le texte ---
    if not os.path.isfile(text_file):
        print(f"❌ Erreur : Le fichier texte '{text_file}' est introuvable.")
        return
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("❌ Erreur : Le fichier texte est vide.")
        return

    # --- 2. Générer l'audio ---
    audio_file = os.path.join(DEFAULT_TEMP_DIR, "audio.mp3")
    clean_directory(DEFAULT_TEMP_DIR)
    generate_audio_from_text(text, audio_file)

    # --- 3. Vérifier le dossier images ---
    if not os.path.isdir(images_dir):
        print(f"❌ Erreur : Le répertoire d'images '{images_dir}' est introuvable.")
        return

    # --- 4. Charger l'audio ---
    print(f"🎵 Chargement de l'audio : {audio_file}")
    audio_clip = AudioFileClip(audio_file)
    video_duration = audio_clip.duration

    # --- 5. Préparer les images ---
    print(f"🖼️  Préparation des images depuis : {images_dir}")
    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print("❌ Erreur : Aucune image trouvée dans le répertoire spécifié.")
        return

    video_size = (1920, 1080)
    resized_images = []
    for img_path in image_files:
        resized_path = resize_image(img_path, video_size, DEFAULT_TEMP_DIR)
        if resized_path:
            resized_images.append(resized_path)
    if not resized_images:
        print("❌ Erreur : Échec du redimensionnement de toutes les images.")
        return

    duration_per_image = video_duration / len(resized_images)
    video_clip = ImageSequenceClip(resized_images, durations=[duration_per_image] * len(resized_images))
    video_clip = video_clip.set_duration(video_duration)

    print(f"✍️  Ajout du texte : '{text}'")
    text_img_path = create_text_image(text, size=video_size, font_size=70)
    text_clip = ImageSequenceClip([text_img_path], durations=[video_duration]).set_duration(video_duration)

    print("✨ Composition de la vidéo finale...")
    final_clip = CompositeVideoClip([video_clip, text_clip])
    final_clip.audio = audio_clip

    print(f"💾 Écriture du fichier : {output_file}")
    try:
        final_clip.write_videofile(
            output_file,
            fps=DEFAULT_FPS,
            codec='libx264',
            audio_codec='aac'
        )
        print(f"🎉 Vidéo générée avec succès : {output_file}")
    except Exception as e:
        print(f"❌ Une erreur est survenue lors de la création de la vidéo : {e}")

    print("🧹 Nettoyage du répertoire temporaire...")
    shutil.rmtree(DEFAULT_TEMP_DIR)
    print("✅ Processus terminé.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs("video/temp", exist_ok=True)
    sections = parse_sections(args.text)
    audio_path = "video/temp/audio.mp3"
    generate_audio_from_sections(sections, audio_path)
    make_video_from_sections(sections, args.images, audio_path, args.output)
