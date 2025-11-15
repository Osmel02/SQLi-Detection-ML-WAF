# 🛡️ SQL Injection Detection with Machine Learning & ModSecurity

## 📋 Descripción del Proyecto

Sistema de detección de inyecciones SQL (SQLi) que combina **Machine Learning** con **ModSecurity** como Web Application Firewall (WAF). La solución utiliza un modelo de ML entrenado para clasificar solicitudes como legítimas o maliciosas, integrado directamente en el flujo de protección del WAF.

## 📁 Estructura del Proyecto

```
SQLi-Detection-ML-WAF/
│
├── 📚 docs/
│   └── Documentación de la implementación.pdf
│
├── 🔬 ml-model/
│   ├── comparativa_de_algoritmos.py
│   ├── SQLi_cleaned_V2.csv
│   └── resultados_modelos_ML/
│       ├── 1-XGBoost.png
│       ├── 2-LightGBM.png
│       ├── 3-LSTM.png
│       ├── 4-CNN.png
│       └── 5-Comparativa_Resultados.png
│
├── ⚙️ waf-config/
│   ├── lua-script/
│   │   └── script.lua
│   ├── modsecurity.conf
│   └── sqli_ml.conf
│
├── 🖥️ flask-server/
│   ├── app.py
│   ├── requirements.txt
│   ├── modelo_entrenado.pkl
│   └── vectorizador.pkl
│
├── 📸 screenshots/
│   ├── logs_modsecurity/
│   ├── pruebas_dvwa/
│   └── resultados_modelos_ML/
│
└── 📄 README.md
```

## 🏗️ Arquitectura del Sistema

```
Cliente → Apache + ModSecurity → Script Lua → Flask + ML Model → Decisión de Bloqueo
```

## 📊 Resultados de Modelos de ML

### Comparativa de Rendimiento

| Modelo | Accuracy | F1-Score | Recall | Tiempo Entrenamiento (s) |
|--------|----------|----------|--------|--------------------------|
| **LightGBM** | **0.9967** | **0.9956** | 0.9939 | **1.63** |
| **XGBoost** | 0.9966 | 0.9954 | 0.9934 | 23.68 |
| **LSTM** | 0.9930 | 0.9906 | **0.9948** | 1333.41 |
| **CNN** | 0.9868 | 0.9820 | 0.9663 | 753.33 |

### 🏆 Modelo Seleccionado: **LightGBM**
- **Mayor precisión general** (99.67%)
- **Tiempo de inferencia más rápido** (1.63 segundos)
- **Ideal para entornos en tiempo real**

## ⚡ Instalación Rápida

### 1. Instalación del Entorno Base
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install apache2 php php-mysql libapache2-mod-php -y
sudo apt install libapache2-mod-security2 -y
```

### 2. Configuración de ModSecurity
```bash
sudo cp /etc/modsecurity/modsecurity.conf-recommended /etc/modsecurity/modsecurity.conf
sudo sed -i 's/SecRuleEngine DetectionOnly/SecRuleEngine On/' /etc/modsecurity/modsecurity.conf
```

### 3. Configuración de Archivos del Proyecto
```bash
# Copiar configuración de WAF
sudo cp waf-config/modsecurity.conf /etc/modsecurity/
sudo cp waf-config/sqli_ml.conf /etc/modsecurity/rules/
sudo cp waf-config/lua-script/script.lua /etc/modsecurity/lua/

# Configurar Flask Server
cd flask-server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Iniciar Servicios
```bash
# Iniciar servidor Flask
cd flask-server
python app.py

# Reiniciar Apache
sudo systemctl restart apache2
```

## 🔧 Configuración

### Reglas ModSecurity
Archivo: `waf-config/sqli_ml.conf`
```apache
SecRule ARGS "@rx .*" \
    "id:1000,\
    phase:2,\
    t:none,\
    deny,\
    status:403,\
    msg:'Ataque SQLi detectado por ML',\
    chain"
    SecRuleScript "/etc/modsecurity/lua/script.lua"
```

### Servidor Flask
El servidor Flask (`flask-server/app.py`) recibe solicitudes desde ModSecurity y las clasifica usando el modelo entrenado.

## 🧪 Pruebas

### Ejemplo de Ataque SQLi
```sql
SELECT * FROM users WHERE id = '1' OR '1'='1'
```

### Verificación de Logs
```bash
tail -f /var/log/apache2/modsec_audit.log
```

## 🛠️ Solución de Problemas

### Error común: Script Lua no ejecuta
```bash
# Verificar permisos
sudo chmod +x /etc/modsecurity/lua/script.lua

# Verificar dependencias Lua
sudo apt install lua-socket lua-json
```

### Modelo no carga correctamente
```bash
# Verificar que los archivos .pkl existan
ls -la flask-server/

# Verificar versión de scikit-learn
pip show scikit-learn
```

## 👨‍💻 Autor

**Osmel Pillot Leyva**  
📅 Proyecto creado el: 01/04/2025

---

⭐ **¿Te gusta este proyecto? Dale una estrella al repositorio!**