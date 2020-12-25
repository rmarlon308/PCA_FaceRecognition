library(shiny)
library(png)
library(dplyr)
library(ggplot2)
library(DT)
library(plotly)
library(jpeg)

images_path = "/home/marlon/mainfolder/marlon/USFQ/DataMining/9_PCA/P9/face_dataset_png"

n_c = nchar(images_path) + 2

all_images = list.files(images_path, pattern = "*.png", full.names = T)

faces_data = NULL

labels = NULL

for(i in all_images){
    sub = substring(i, n_c)
    labels = c(labels, substring(sub, 1, 2))
    image_vector = as.vector(readPNG(i))
    faces_data = rbind(faces_data, image_vector) #By rows
}

#Plot Images function
plotImage = function(image_vector, title){
    size = sqrt(length(image_vector))
    dim(image_vector) = c(size, size)
    image_vector = t(apply(image_vector, 2, rev))
    image(image_vector, col = grey(seq(0, 1, length = 256)), main = title)
}

training = faces_data

#Average Face
mean_face = colMeans(training)

#Center Data
normalized_data = training
for(i in 1: nrow(normalized_data)){
    normalized_data[i, ] = normalized_data[i,] - mean_face
}

#Covariance Matrix
svd = svd(normalized_data)

eigenVectors = svd$v
eigenValues = svd$d ^ 2 / sum(svd$d ^ 2)

cumVariance = cumsum(eigenValues) / sum(eigenValues)

k = NULL

for(i in 1:length(cumVariance)){
    if(cumVariance[i] >= 0.95){ #Take the 95% of the variance
        k = i
        break
    }
}

scree_data = data.frame("pc" = 1:k, "eigenValues" = eigenValues[1:k])



ui <- fluidPage(
    
    titlePanel("PCA Face Recognition"),
    
    sidebarLayout(
        sidebarPanel(
            sliderInput("range", "Choose a K: ",min = 1, max = k,value = k),
            width = 3
        ),
        
        mainPanel(
            
            tabsetPanel(
                type = "tabs",
                
                tabPanel("Mean Face", plotOutput("meanFacePlot", height = "700px")),
                tabPanel("Scree Plot", plotlyOutput("screePlot")),
                tabPanel("Variance", textOutput("kValue")),
                tabPanel("Eigenvectors", 
                         uiOutput("eigenFace"),
                         plotOutput("eigenPlot", height = "700px")
                ),
                tabPanel("Matrix", DT::dataTableOutput("table")),
                tabPanel("Face Recognition",
                         fileInput("file", "Chose one or multiple Images", accept = ".png", multiple = T),
                         uiOutput("test_faces"),
                         plotOutput("chosen_image"),
                         plotOutput("recognized")
                ),
                tags$head(
                    tags$style(
                        "#kValue{font-size: 20px}"
                    )
                )
                
            )
            
            
        )
    )
)

server <- function(input, output, session) {
    
    k = reactiveValues(k_selected = 1)
    
    file = reactive({
        input$file
    })
    
    image_path = reactive({
        file()$datapath
    })
    
    image_header = reactive({
        file()$name
    })
    
    
    output$test_faces = renderUI({
        n = length(image_path())
        sliderInput("nTest", "Choose a  test Face: ", min = 1, max = n, value = 1, step = 1)
    })
    
    image_vector = reactive({
        image_number = input$nTest
        
        as.vector(readPNG(image_path()[image_number]))
    })
    
    image_name = reactive({
        image_number = input$nTest
        
        image_header()[image_number]
    })
    
    training_weights = reactive({
        as.data.frame(normalized_data %*% eigenVectors[,1:k$k_selected])
        
    })
    
    output$meanFacePlot = renderPlot({
        plotImage(mean_face, "Mean Face")
    })
    
    output$screePlot = renderPlotly({
        scree = ggplot(scree_data, aes(x = pc, y = eigenValues)) + 
            geom_line(color = "grey")+
            geom_point(color = "blue",size =2)+
            scale_x_continuous(labels = as.character(1:nrow(scree_data)), breaks = 1:nrow(scree_data))+
            xlab("Principal Component") + 
            ylab("Variance")+
            labs(title = "Screeplot")
        
        ggplotly(scree)
    })
    
    observe({
        k$k_selected = input$range
    })
    
    
    output$kValue = renderText({
        paste("Variance: ", cumVariance[k$k_selected])
    })
    
    output$eigenFace = renderUI({
        sliderInput("nFace", "Choose a Face: ", min = 1, max = k$k_selected, value = 1, step = 1)
    })
    
    output$eigenPlot = renderPlot({
        plotImage(eigenVectors[, input$nFace], paste("Eigenface", input$nFace))
    })
    
    output$table = DT::renderDataTable(
        training_weights(),extensions =c("ColReorder","FixedColumns"), options = list(
            scrollX = TRUE,
            colReorder = TRUE,
            fixedColumns = list(leftColumns = 1,leftColumns = 3),
            pageLength=10
        )
    )
    
    
    output$chosen_image = renderPlot({
        plotImage(image_vector(), image_name())
    })
    
    output$recognized = renderPlot({
        normalized_image = image_vector() - mean_face
        
        test_weights = as.vector(t(eigenVectors[, 1:k$k_selected]) %*% normalized_image)
        
        euclidean_distance = NULL
        position = NULL
        
        for(i in 1:nrow(training_weights())){
            euclidean_distance[i] = dist(rbind(training_weights()[i,], test_weights))
            position[i] = i
        }
        dist = data.frame(position, euclidean_distance) %>%
            arrange(euclidean_distance)
        
        col = dist[1,1]
        
        final_image = training_weights()[col,]
        
        im = as.vector((as.matrix(final_image) %*% as.matrix(t(eigenVectors[, 1:k$k_selected]))) + mean_face)
        
        plotImage(im, paste("Recognized:", labels[col]))
        
    })

}

shinyApp(ui = ui, server = server)