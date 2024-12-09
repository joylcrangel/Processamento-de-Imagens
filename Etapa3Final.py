import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np  # Manipular manualmente matriz dos filtros

# Variáveis globais
img_cv = None  # Imagem original carregada
segmentation_result = None  # Resultado da segmentação

def open_main_window():
    # Fecha a janela inicial
    start_window.destroy()

    # Cria a janela principal
    root = tk.Tk()
    root.title("Image Processing App")
    root.geometry("1300x630")
    root.config(bg="#2e2e2e")
    root.resizable(False, False)  # Impede o redimensionamento da janela principal

    # Carregar o ícone da janela (substitua 'icon_image.png' pelo caminho da sua imagem de ícone)
    icon_img = ImageTk.PhotoImage(file="icon_image.png")
    root.iconphoto(False, icon_img)  # Definir o ícone da janela

    # Carregar a imagem de fundo
    bg_imageEd = Image.open("Fundo Editor.png")
    bg_imageEd = bg_imageEd.resize((1300, 630))
    bg_imageEd_tk = ImageTk.PhotoImage(bg_imageEd)

    # Criar um Label com a imagem de fundo
    bg_labelEd = tk.Label(root, image=bg_imageEd_tk)
    bg_labelEd.place(x=0, y=0, relwidth=1, relheight=1)

    img_cv = None

    def load_image():
        nonlocal img_cv
        file_path = filedialog.askopenfilename()
        if file_path:
            img_cv = cv2.imread(file_path)
            display_image(img_cv, original=True)

    def display_image(img, original=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((500, 500))  # Mantém a proporção da imagem
        img_tk = ImageTk.PhotoImage(img_pil)

        x_offset = (500 - img_pil.width) // 2
        y_offset = (500 - img_pil.height) // 2

        if original:
            original_image_canvas.delete("all")
            original_image_canvas.image = img_tk
            original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
        else:
            edited_image_canvas.delete("all")
            edited_image_canvas.image = img_tk
            edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)


########3#######IMAGEM EM ESCALA DE CINZAS##########################

    def convert_to_grayscale(img):
        """
        Converte uma imagem para escala de cinza, se necessário.
        :param img: Imagem de entrada.
        :return: Imagem em escala de cinza.
        """
        if len(img.shape) == 3:  # Verifica se a imagem tem 3 canais (colorida)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img  # Retorna a imagem original se já for em escala de cinza

##########################FILTROS###################################

    # Dicionário para salvar os valores dos sliders
    slider_values = {
        "gaussian": 0,
        "mean": 0,
        "sobel": 0,
        "laplacian": 0
    }

    def gaussian_filter(value):
        """Atualiza o valor do filtro Gaussiano e aplica os filtros."""
        slider_values["gaussian"] = int(value)  # Converter para inteiro
        update_pipeline()

    def mean_filter(value):
        """Atualiza o valor do filtro Média e aplica os filtros."""
        slider_values["mean"] = int(value)  # Converter para inteiro
        update_pipeline()

    def sobel_filter(value):
        """Atualiza o valor do filtro Sobel e aplica os filtros."""
        slider_values["sobel"] = int(value)  # Converter para inteiro
        update_pipeline()

    def laplacian_filter(value):
        """Atualiza o valor do filtro Laplaciano e aplica os filtros."""
        slider_values["laplacian"] = int(value)  # Converter para inteiro
        update_pipeline()


#------------------------OPERAÇÕES MORFOLOGICAS---------------------------

    def manual_threshold(img, threshold_value):
        """Aplica limiarização manual."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def otsu_threshold(img):
        """Aplica limiarização usando Otsu."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        otsu_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), otsu_value  # Retorna o valor de Otsu também

    def apply_morphology(img, kernel_size, operation):
        """Aplica operações morfológicas."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if operation == "Erosion":
            morphed = cv2.erode(gray, kernel)
        elif operation == "Dilation":
            morphed = cv2.dilate(gray, kernel)
        elif operation == "Opening":
            morphed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elif operation == "Closing":
            morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        else:
            return img  # Retorna a imagem original se a operação não for definida
        return cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)

########################Função Cumulativa Filtros############################################

    


########################Função Cumulativa de Morfologia\Segmentação##########################

    def update_pipeline():
        """Atualiza a imagem processada com base nos controles."""
        if img_cv is None:
            return

        # INÍCIO COM IMAGEM ORIGINAL
        temp_img = img_cv.copy()

        # FILTROS ACUMULATIVOS
        if slider_values["gaussian"] > 0:
            kernel_size = int(slider_values["gaussian"]) | 1  # Kernel deve ser ímpar
            temp_img = cv2.GaussianBlur(temp_img, (kernel_size, kernel_size), 0)

        if slider_values["mean"] > 0:
            kernel_size = int(slider_values["mean"]) | 1
            temp_img = cv2.blur(temp_img, (kernel_size, kernel_size))

        if slider_values["sobel"] > 0:
            kernel_size = int(slider_values["sobel"]) | 1
            gray_img = convert_to_grayscale(temp_img)
            sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
            sobel_magnitude = cv2.magnitude(sobelx, sobely)
            temp_img = np.uint8(cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX))
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2RGB)

        if slider_values["laplacian"] > 0:
            kernel_size = int(slider_values["laplacian"]) | 1
            gray_img = convert_to_grayscale(temp_img)
            laplacian = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=kernel_size)
            laplacian_abs = cv2.convertScaleAbs(laplacian)
            temp_img = np.uint8(cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX))
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2RGB)

        # SEGMENTAÇÃO
        if segmentation_mode.get() == "Manual":
            threshold_slider.config(state="normal")
            temp_img = manual_threshold(temp_img, threshold_slider.get())
            otsu_value_label.config(text="Otsu: N/A")

        elif segmentation_mode.get() == "Otsu":
            threshold_slider.config(state="disabled")
            temp_img, otsu_value = otsu_threshold(temp_img)
            otsu_value_label.config(text=f"Otsu: {int(otsu_value)}")

        else:
            threshold_slider.config(state="disabled")
            otsu_value_label.config(text="Otsu: N/A")

        # OPERAÇÕES MORFOLÓGICAS
        kernel_size = morph_slider.get() | 1
        temp_img = apply_morphology(temp_img, kernel_size, morph_mode.get())

        # ATUALIZA IMAGEM PROCESSADA
        global processed_img
        processed_img = temp_img
        display_image(processed_img, original=False)


################## RESETAR IMAGEM ##############################

    # Variável para armazenar a imagem processada
    processed_img = None  # Inicialmente nenhuma imagem foi processada

    # Função para resetar a imagem processada e os filtros
    def reset_processed_image():
        """
        Reseta a imagem processada, voltando à original e resetando os filtros aplicados.
        """
        nonlocal processed_img
        processed_img = None  # Limpa a imagem processada
        
        # Resetando os valores dos sliders para 0 (sem filtros aplicados)
        slider_values["gaussian"] = 0
        slider_values["mean"] = 0
        slider_values["sobel"] = 0
        slider_values["laplacian"] = 0

        # Reseta os sliders para o valor inicial
        blur_slider.set(0)
        mean_slider.set(0)
        sobel_slider.set(0)
        laplacian_slider.set(0)

        # Atualiza a exibição da imagem original
        display_image(img_cv, original=False)


#############################################################################

    # Criação do menu principal
    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)

    # Menu para carregar e sair do programa
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Load Image", command=load_image)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # Adicionando o reset na barra de menu
    menu_bar.add_cascade(label="Reset Image", command=reset_processed_image)
    #file_menu.add_command(label="Reset Image", command=reset_processed_image)

    # Criação dos Canvas para exibição das imagens
    original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
    original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

    edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
    edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

    #### Painel de controle ###
    #GAUSS
    control_frame = tk.Frame(root, bg="#454546")
    control_frame.grid(row=0, column=2, padx=20, pady=20, sticky="n")

    tk.Label(control_frame, text="Gaussian Blur", bg="#454546", fg="white").pack(pady=5)
    blur_slider = tk.Scale(control_frame, from_=1, to=31, resolution=2, orient="horizontal", command=gaussian_filter, bg="#8c8c8c", fg="white")
    blur_slider.pack()

    #MÉDIA
    tk.Label(control_frame, text="Mean Filter", bg="#454546", fg="white").pack(pady=5)
    mean_slider = tk.Scale(control_frame, from_=1, to=31, resolution=2, orient="horizontal", command=mean_filter, bg="#8c8c8c", fg="white")
    mean_slider.pack()
    
    #SOBEL
    tk.Label(control_frame, text="Sobel Filter", bg="#454546", fg="white").pack(pady=5)
    sobel_slider = tk.Scale(control_frame, from_=1, to=31, resolution=2, orient="horizontal", command=sobel_filter, bg="#8c8c8c", fg="white")
    sobel_slider.pack()

    #LAPLACIANO
    tk.Label(control_frame, text="Laplacian Filter", bg="#454546", fg="white").pack(pady=5)
    laplacian_slider = tk.Scale(control_frame, from_=1, to=31, resolution=2, orient="horizontal", command=laplacian_filter, bg="#8c8c8c", fg="white")
    laplacian_slider.pack()

    #### Painel de controle ###
    # Frame para segmentação
    segmentation_mode = tk.StringVar(value="None")
    tk.Label(control_frame, text="Segmentação").pack(pady=5)
    segmentation_dropdown = ttk.Combobox(control_frame, textvariable=segmentation_mode, state="readonly",
                                        values=["None", "Manual", "Otsu"])
    segmentation_dropdown.pack()
    segmentation_dropdown.bind("<<ComboboxSelected>>", lambda e: update_pipeline())

    # Slider de limiar manual
    threshold_slider = tk.Scale(control_frame, from_=0, to=255, orient="horizontal",
                                label="Limiar (Manual)", command=lambda e: update_pipeline())
    threshold_slider.pack()

    # Label para exibir o valor do limiar de Otsu
    otsu_value_label = tk.Label(control_frame, text="Otsu: N/A")
    otsu_value_label.pack(pady=5)

    # Frame para morfologia
    morph_mode = tk.StringVar(value="None")
    tk.Label(control_frame, text="Morfologia").pack(pady=5)
    morph_dropdown = ttk.Combobox(control_frame, textvariable=morph_mode, state="readonly",
                                values=["None", "Erosion", "Dilation", "Opening", "Closing"])
    morph_dropdown.pack()
    morph_dropdown.bind("<<ComboboxSelected>>", lambda e: update_pipeline())

    morph_slider = tk.Scale(control_frame, from_=1, to=31, resolution=2, orient="horizontal",
                            label="Tamanho do Kernel", command=lambda e: update_pipeline())
    morph_slider.pack()


    root.mainloop()


######################################################################################

# Janela inicial
start_window = tk.Tk()
start_window.title("Bem-vindo ao B&J")
start_window.geometry("960x540")
start_window.config(bg="#2e2e2e")
start_window.resizable(False, False)  # Impede o redimensionamento da janela inicial

# Carregar o ícone da janela inicial (substitua 'icon_image.png' pelo caminho da sua imagem de ícone)
icon_img = ImageTk.PhotoImage(file="icon_image.png")
start_window.iconphoto(False, icon_img)  # Definir o ícone da janela inicial

# Carregar a imagem de fundo
bg_image = Image.open("Bem vindo.png")
bg_image = bg_image.resize((960, 540))
bg_image_tk = ImageTk.PhotoImage(bg_image)

# Criar um Label com a imagem de fundo
bg_label = tk.Label(start_window, image=bg_image_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Criar um botão "Comece Agora"
start_button = tk.Button(start_window, text="Comece Agora!", font=("Helvetica", 14), bg="#E67E22", fg="white", command=open_main_window)
start_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

start_window.mainloop()
