import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np  # Manipular manualmente matriz dos filtros


def open_main_window():
    # Fecha a janela inicial
    start_window.destroy()

    # Cria a janela principal
    root = tk.Tk()
    root.title("Image Processing App")
    root.geometry("1300x600")
    root.config(bg="#2e2e2e")
    root.resizable(False, False)  # Impede o redimensionamento da janela principal

    # Carregar o ícone da janela (substitua 'icon_image.png' pelo caminho da sua imagem de ícone)
    icon_img = ImageTk.PhotoImage(file="icon_image.png")
    root.iconphoto(False, icon_img)  # Definir o ícone da janela

    # Carregar a imagem de fundo
    bg_imageEd = Image.open("Fundo Editor.png")
    bg_imageEd = bg_imageEd.resize((1300, 600))
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
        Converte uma imagem RGB para escala de cinza manualmente.
        :param img: Imagem de entrada (RGB).
        :return: Imagem em escala de cinza.
        """
        # Verifica se a imagem está no formato RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Extrai os canais R, G e B
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            # Calcula a escala de cinza usando a fórmula ponderada
            gray_img = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
            return gray_img
        else:
            # Se a imagem já está em escala de cinza, retorna como está
            return img

##########################FILTROS###################################

    def gaussian_kernel(size, sigma=1.0):
        """
        Cria um kernel Gaussiano 2D.
        
        :param size: Tamanho do kernel (deve ser ímpar).
        :param sigma: Desvio padrão da função Gaussiana.
        :return: Kernel Gaussiano 2D.
        """
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        sum_val = 0  # Para normalizar o kernel depois

        # Preenchendo o kernel com valores baseados na equação gaussiana
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                sum_val += kernel[i, j]

        # Normalizando o kernel para que a soma dos valores seja 1
        kernel /= sum_val
        return kernel

    def gaussian_filter(img):
        """
        Aplica um filtro Gaussiano manualmente a uma imagem colorida.
        
        :param img: Imagem de entrada (assumida como np.uint8).
        :return: Imagem filtrada.
        """
        gaussian_kernel = np.array([[1,  4,  6,  4,  1],
                                            [4, 20, 30, 20,  4],  # Aumentado para 20 e 30
                                            [6, 30, 50, 30,  6],  # Aumentado para 30 e 50
                                            [4, 20, 30, 20,  4],
                                            [1,  4,  6,  4,  1]], dtype=np.float32)
                
        # Normalizando o kernel (soma deve ser 1)
        gaussian_kernel /= np.sum(gaussian_kernel)

        # Aplicar o filtro Gaussiano em cada canal da imagem colorida
        channels = cv2.split(img)  # Separar a imagem em seus três canais (R, G, B)
        filtered_channels = [apply_kernel(channel, gaussian_kernel) for channel in channels]
        
        # Juntar os canais novamente
        filtered_img = cv2.merge(filtered_channels)
        
        return np.uint8(filtered_img)


    def mean_filter(img, kernel_size):
        """
        Aplica um filtro de média manualmente a uma imagem.
        
        :param img: Imagem de entrada (assumida como np.uint8).
        :param kernel_size: Tamanho da janela do filtro (deve ser ímpar).
        :return: Imagem filtrada.
        """
        half_k = kernel_size // 2
        rows, cols = img.shape[:2]
        output = np.zeros_like(img)

        for i in range(rows):
            for j in range(cols):
                # Definindo os limites da região do kernel
                x_min = max(i - half_k, 0)
                x_max = min(i + half_k + 1, rows)
                y_min = max(j - half_k, 0)
                y_max = min(j + half_k + 1, cols)

                # Extração da região
                region = img[x_min:x_max, y_min:y_max]

                # Cálculo da média
                output[i, j] = np.mean(region, axis=(0, 1))

        return output.astype(np.uint8)  # Garantindo que a saída esteja no formato correto

    def apply_kernel(img, kernel):
        """
        Aplica um kernel a uma imagem de forma manual.
        
        :param img: Imagem de entrada em escala de cinza.
        :param kernel: Kernel a ser aplicado.
        :return: Imagem filtrada.
        """
        img_height, img_width = img.shape
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2

        padded_img = np.pad(img, pad, mode='constant', constant_values=0)
        filtered_img = np.zeros_like(img, dtype=np.float32)

        for i in range(img_height):
            for j in range(img_width):
                region = padded_img[i:i+kernel_size, j:j+kernel_size]
                filtered_img[i, j] = np.sum(region * kernel)
        
        return filtered_img

    def sobel_filter(gray_img):
        """
        Aplica um filtro Sobel manualmente a uma imagem em escala de cinza.
        :param gray_img: Imagem de entrada já em escala de cinza (np.uint8).
        :return: Imagem filtrada.
        """
        # Definir os kernels Sobel
        sobelx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobely_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Aplicar os kernels Sobel em X e Y
        sobelx = apply_kernel(gray_img, sobelx_kernel)
        sobely = apply_kernel(gray_img, sobely_kernel)

        # Calcular a magnitude do gradiente
        sobel_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))

        # Normalizar a magnitude para o intervalo [0, 255]
        sobel_magnitude = 255 * (sobel_magnitude - np.min(sobel_magnitude)) / (np.max(sobel_magnitude) - np.min(sobel_magnitude))
        
        # Aumentar o contraste
        sobel_magnitude = np.clip(sobel_magnitude * 4, 0, 255)

        # Aplicar um limiar para realçar áreas de borda
        threshold = 1000  # Ajuste esse valor conforme necessário
        sobel_magnitude = np.where(sobel_magnitude > threshold, 255, sobel_magnitude)

        return np.uint8(sobel_magnitude)

    def laplacian_filter(gray_img):
        """
        Aplica um filtro Laplaciano manualmente a uma imagem em escala de cinza.
        :param gray_img: Imagem de entrada já em escala de cinza (np.uint8).
        :return: Imagem filtrada.
        """
        # Definir o kernel Laplaciano para detecção de bordas
        laplacian_kernel = np.array([[0, 1, 0], 
                                    [1, -4, 1], 
                                    [0, 1, 0]], dtype=np.float32)
        
        # Aplicar o kernel Laplaciano
        laplacian = apply_kernel(gray_img, laplacian_kernel)

        # Amplificar os contornos
        amplification_factor = 2.0
        laplacian = laplacian * amplification_factor

        # Clipping para garantir valores no intervalo [0, 255]
        laplacian = np.clip(laplacian, 0, 255)

        return np.uint8(laplacian)

#------------------------OPERAÇÕES MORFOLOGICAS---------------------------

    #Erosão
    def erode_image(img, kernel_size=3):
        """
        Aplica erosão manualmente a uma imagem.
        :param img: Imagem de entrada em escala de cinza.
        :param kernel_size: Tamanho do kernel (deve ser ímpar).
        """
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        pad = kernel_size // 2
        padded_img = np.pad(img, pad, mode='constant', constant_values=0)
        eroded_img = np.zeros_like(img)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded_img[i:i+kernel_size, j:j+kernel_size]
                eroded_img[i, j] = np.min(region * kernel)
        
        return eroded_img
    
    #Dilatação
    def dilate_image(img, kernel_size=3):
        """
        Aplica dilatação manualmente a uma imagem.
        :param img: Imagem de entrada em escala de cinza.
        :param kernel_size: Tamanho do kernel (deve ser ímpar).
        """
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        pad = kernel_size // 2
        padded_img = np.pad(img, pad, mode='constant', constant_values=0)
        dilated_img = np.zeros_like(img)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded_img[i:i+kernel_size, j:j+kernel_size]
                dilated_img[i, j] = np.max(region * kernel)
        
        return dilated_img

    #Abertura e fechamento
    def opening(img, kernel_size=3):
        """
        Aplica abertura (erosão seguida de dilatação).
        """
        eroded = erode_image(img, kernel_size)
        opened = dilate_image(eroded, kernel_size)
        return opened

    def closing(img, kernel_size=3):
        """
        Aplica fechamento (dilatação seguida de erosão).
        """
        dilated = dilate_image(img, kernel_size)
        closed = erode_image(dilated, kernel_size)
        return closed

    #Limiarização (Thresholding)
    def manual_threshold(gray_img, threshold_value):
        """
        Aplica uma limiarização binária manualmente a uma imagem em escala de cinza.
        :param gray_img: Imagem de entrada já em escala de cinza (np.uint8).
        :param threshold_value: Valor do limiar.
        :return: Imagem limiarizada (binária).
        """
        # Inicializa uma matriz para a imagem limiarizada
        thresholded_img = np.zeros_like(gray_img, dtype=np.uint8)

        # Percorre todos os pixels da imagem e aplica o limiar
        rows, cols = gray_img.shape
        for i in range(rows):
            for j in range(cols):
                if gray_img[i, j] > threshold_value:
                    thresholded_img[i, j] = 255  # Branco
                else:
                    thresholded_img[i, j] = 0    # Preto

        return thresholded_img


    #Limiarização Adaptativa (Otsu)
    def otsu_threshold(gray_img):
        """
        Implementação manual do método de limiarização de Otsu.
        :param gray_img: Imagem de entrada já em escala de cinza (np.uint8).
        :return: Imagem binária após a aplicação da limiarização de Otsu.
        """
        # Calcula o histograma da imagem
        histogram = np.histogram(gray_img, bins=256, range=(0, 256))[0]
        total_pixels = gray_img.size

        current_max, threshold_value = 0, 0
        sum_total, sum_background = 0, 0
        weight_background, weight_foreground = 0, 0

        for i in range(256):
            sum_total += i * histogram[i]

        for i in range(256):
            weight_background += histogram[i]
            if weight_background == 0:
                continue
            
            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break
            
            sum_background += i * histogram[i]
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground

            # Calcula a variância entre classes
            between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

            # Verifica se essa é a maior variância entre classes encontrada
            if between_class_variance > current_max:
                current_max = between_class_variance
                threshold_value = i

        # Aplica o limiar de Otsu encontrado
        return manual_threshold(gray_img, threshold_value)

################## APLICAR FILTROS E MORFOLOGIA ##############################

    def apply_filter(filter_type, threshold_type='manual'):
        """
        Função para aplicar os filtros escolhidos.
        :param filter_type: Tipo de filtro ou operação morfológica a ser aplicada.
        :param threshold_type: Tipo de limiarização ('manual' ou 'otsu').
        """
        if img_cv is None:
            return
    
        # Converte para escala de cinza manualmente
        gray = convert_to_grayscale(img_cv)

        # Filtros de suavização e detecção de bordas
        if filter_type == "low_pass":
            filtered_img = gaussian_filter(img_cv)
        elif filter_type == "mean":
            filtered_img = mean_filter(img_cv, kernel_size=3)
        elif filter_type == "high_pass":
            filtered_img = laplacian_filter(gray)
        elif filter_type == "sobel":
            filtered_img = sobel_filter(gray)

        # Operações morfológicas
        elif filter_type == "erosion threshold":
            # Aplicar erosão manual
            binary_img = manual_threshold(gray, 127)
            filtered_img = erode_image(binary_img)
        elif filter_type == "erosion otsu":
            # Aplicar erosão manual
            binary_img = otsu_threshold(gray)
            filtered_img = erode_image(binary_img)  
        #----------------------------------------------------
        elif filter_type == "dilation threshold":
            # Aplicar dilatação manual
            binary_img = manual_threshold(gray, 127)
            filtered_img = dilate_image(binary_img)
        elif filter_type == "dilation otsu":
            # Aplicar dilatação manual
            binary_img = otsu_threshold(gray)
            filtered_img = dilate_image(binary_img)
        #----------------------------------------------------
        elif filter_type == "opening threshold":
            # Aplicar abertura (erosão seguida de dilatação)
            binary_img = manual_threshold(gray, 127)
            filtered_img = opening(binary_img)
        elif filter_type == "opening otsu":
            # Aplicar abertura (erosão seguida de dilatação)
            binary_img = otsu_threshold(gray)
            filtered_img = opening(binary_img)
        #----------------------------------------------------
        elif filter_type == "closing threshold":
            # Aplicar fechamento (dilatação seguida de erosão)
            binary_img = manual_threshold(gray, 127)
            filtered_img = closing(binary_img)
        elif filter_type == "closing otsu":
            # Aplicar fechamento (dilatação seguida de erosão)
            binary_img = otsu_threshold(gray)
            filtered_img = closing(binary_img)

        # Exibir a imagem editada
        display_image(filtered_img, original=False)

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

    # Menu de Filtros (Passa-baixa e Passa-alta)
    filters_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Filters", menu=filters_menu)
    filters_menu.add_command(label="Low Pass Filter (Gaussian)", command=lambda: apply_filter("low_pass"))
    filters_menu.add_command(label="Low Pass Filter (Mean)", command=lambda: apply_filter("mean"))
    filters_menu.add_command(label="High Pass Filter (Laplacian)", command=lambda: apply_filter("high_pass"))
    filters_menu.add_command(label="High Pass Filter (Sobel)", command=lambda: apply_filter("sobel"))

    # Menu de Operações Morfológicas
    morphology_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Morphology", menu=morphology_menu)
    morphology_menu.add_command(label="Erosion (Threshold)", command=lambda: apply_filter("erosion threshold"))
    morphology_menu.add_command(label="Erosion (Otsu)", command=lambda: apply_filter("erosion otsu"))
    morphology_menu.add_command(label="Dilation (Threshold)", command=lambda: apply_filter("dilation threshold"))
    morphology_menu.add_command(label="Dilation (Otsu)", command=lambda: apply_filter("dilation otsu"))
    morphology_menu.add_command(label="Opening (Threshold)", command=lambda: apply_filter("opening threshold"))
    morphology_menu.add_command(label="Opening (Otsu)", command=lambda: apply_filter("opening otsu"))
    morphology_menu.add_command(label="Closing (Threshold)", command=lambda: apply_filter("closing threshold"))
    morphology_menu.add_command(label="Closing (Otsu)", command=lambda: apply_filter("closing otsu"))

    # Criação dos Canvas para exibição das imagens
    original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
    original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

    edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
    edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

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
