# app.py
import os
import time
import google.generativeai as genai
from rich import print
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
import joblib
import numpy as np

# Configuración de Gemini 2.0
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

class AISecurityTrainer:
    def __init__(self):
        self.model = None
        self.training_data = None
        self.load_model()
        
    def load_model(self):
        try:
            self.model = joblib.load("wildpass_model.pkl")
            print(Panel("[green]✅ Modelo IA cargado con éxito[/green]", border_style="green"))
        except:
            self.model = None
            
    def generate_training_data(self, samples=500):
        weak_passwords = []
        strong_passwords = []
        
        # Generar contraseñas débiles con Gemini
        response = model.generate_content(
            "Genera 250 ejemplos de contraseñas inseguras comunes, solo las contraseñas separadas por comas"
        )
        weak_passwords = response.text.split(", ")[:250]
        
        # Generar contraseñas fuertes con Gemini
        response = model.generate_content(
            "Genera 250 ejemplos de contraseñas seguras complejas, solo las contraseñas separadas por comas"
        )
        strong_passwords = response.text.split(", ")[:250]
        
        X = [self.extract_features(pwd) for pwd in weak_passwords + strong_passwords]
        y = [0]*len(weak_passwords) + [1]*len(strong_passwords)
        
        return np.array(X), np.array(y)
    
    def extract_features(self, password):
        return [
            len(password),
            sum(c.isupper() for c in password),
            sum(c.isdigit() for c in password),
            sum(c in string.punctuation for c in password),
            len(set(password))/max(len(password), 1)
        ]
    
    def dynamic_train(self):
        with Live(self.create_loading_panel(0), refresh_per_second=10, screen=True) as live:
            # Generación de datos con animación
            for i in range(1, 101):
                time.sleep(0.05)
                live.update(self.create_loading_panel(i))
                
            X, y = self.generate_training_data()
            
            # Entrenamiento con actualizaciones visuales
            self.model = RandomForestClassifier(n_estimators=100, warm_start=True)
            accuracies = []
            
            for epoch in range(1, 101):
                self.model.n_estimators = epoch
                self.model.fit(X, y)
                acc = self.model.score(X, y)
                accuracies.append(acc)
                
                live.update(self.create_training_panel(epoch, acc, accuracies))
                time.sleep(0.05)
                
            joblib.dump(self.model, "wildpass_model.pkl")
            
    def create_training_panel(self, epoch, accuracy, history):
        grid = Table.grid(expand=True)
        grid.add_column("Training", justify="left")
        grid.add_column("Metrics", justify="right")
        
        grid.add_row(
            f"[bold]Época:[/bold] {epoch}/100\n"
            f"[chart]▏{'█' * int(accuracy*50)}{'░' * (50 - int(accuracy*50))}[/chart]",
            
            f"[bold]Precisión:[/bold] {accuracy*100:.1f}%\n"
            f"[bold]Árboles:[/bold] {epoch*10}\n"
            f"[bold]Muestras:[/bold] {len(X) if 'X' in locals() else 0}"
        )
        
        return Panel(
            grid,
            title="[bold blue]WildPassPro - Entrenamiento con Gemini 2.0[/]",
            border_style="blue",
            padding=(1, 2)
        )
        
    def create_loading_panel(self, progress):
        phases = [
            "🔍 Analizando patrones globales...",
            "🛡️ Optimizando seguridad...",
            "🌐 Consultando red neuronal...",
            "🧠 Procesando con Gemini..."
        ]
        return Panel(
            f"{phases[progress % 4]}\n\n"
            f"[cyan]{'▉' * (progress % 10)}{' ' * (10 - (progress % 10))}[/]",
            title="[bold]Inicializando IA...[/]",
            border_style="cyan"
        )

def main_interface():
    print(Panel.fit("[bold red]🔥 WILDPASS PRO - SEGURIDAD CUÁNTICA 🔥[/]", border_style="red"))
    
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    
    trainer = AISecurityTrainer()
    
    while True:
        layout["header"].update(
            Panel("[bold]1. Generar Contraseña Cuántica\n2. Escáner de Seguridad\n3. Entrenamiento en Vivo\n4. Salir[/]", 
                 border_style="yellow"))
        
        choice = input("Selección: ")
        
        if choice == "3":
            trainer.dynamic_train()
        elif choice == "4":
            break

if __name__ == "__main__":
    main_interface()
