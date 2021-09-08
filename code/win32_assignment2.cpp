#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h> 

#include "glad/glad.h"
#include "glad/glad.c"

#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"

//IMGUI
#include "imgui/imgui.cpp"
#include "imgui/imgui_draw.cpp"
#include "imgui/imgui_widgets.cpp"
#include "imgui/imgui_impl_glfw.cpp"
#include "imgui/imgui_impl_opengl3.cpp"
#include "imgui/imgui_demo.cpp"
#include "imgui/imconfig.h"

#include "utils.h"

global u64 g_PerformanceCountFreq;
#include "win32_assignment2_string.cpp"
#include "win32_memory.cpp"
#include "win32_assignment2_data.cpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	
    glViewport(0, 0, width, height);
} 

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}


int main() {
	LARGE_INTEGER PerfCountFrequencyResult;
    QueryPerformanceFrequency(&PerfCountFrequencyResult);
	g_PerformanceCountFreq = PerfCountFrequencyResult.QuadPart;
	
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	GLFWwindow* window = glfwCreateWindow(800,600, "Assignment2", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  
	glfwSwapInterval(1);
	
	if(window == NULL) {
		glfwTerminate();
		return -1;
	}
	
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		return -1;
	}  
	
	//IMGUI init
	IMGUI_CHECKVERSION();
    ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 150");
	bool show_window = true;
	
	//MemoryArena's
	g_MemoryInfo.permanent_arena = Memory_ArenaInitialise();
	g_MemoryInfo.file_arena = Memory_ArenaInitialise();
	g_MemoryInfo.temp_calc_arena = Memory_ArenaInitialise();
	g_MemoryInfo.calc_arena = Memory_ArenaInitialise();
	
	OPENFILENAME ofn = {0}; 
	TCHAR szFile[260]={0};
	// Initialize remaining fields of OPENFILENAME structure
	ofn.lStructSize = sizeof(ofn); 
	ofn.hwndOwner = glfwGetWin32Window(window); 
	ofn.lpstrFile = szFile; 
	ofn.nMaxFile = sizeof(szFile); 
	ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0"; 
	ofn.nFilterIndex = 1; 
	ofn.lpstrFileTitle = NULL; 
	ofn.nMaxFileTitle = 0; 
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	
	b32 file_is_read = false;
	
	FileRead file_read = {};
	ParsedFile parsed_file = {};
	while(!glfwWindowShouldClose(window))
	{
		
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::Begin("SVM Machine Learning Assignment 2", &show_window, NULL);
		ImGui::Text("Open a Tab Delimited File");
		static b32  test_completed = false;
		
		if(ImGui::Button("Open")) {
			if(GetOpenFileName(&ofn) == TRUE)
			{ 
				Memory_ArenaClear(&g_MemoryInfo.calc_arena);
				Memory_ArenaClear(&g_MemoryInfo.file_arena);
				
				file_read = Win32_ReadEntireFile(ofn.lpstrFile);
				parsed_file = ParseFile(file_read);
				file_is_read = true;
				test_completed = false;
				
				
			}
		}
		ImGui::Text("Open File: %s", ofn.lpstrFile);
		
		static int current_col = 0;
		static float test_split = 0.5f;
		static int num_iterations = 100;
		static float gainC = 100.0f;
		static SVMModel *model = NULL;
		static SVMTestResults *test_results;
		static f32 time_taken = 0;
		
		if(file_is_read) {
			ImGui::Text("Select the Label Column");
			ImGui::ListBox("Attributes", &current_col, parsed_file.header->entries, parsed_file.num_cols, 8);
			ImGui::Text("Selected: %s", parsed_file.header->entries[current_col]);
			
			ImGui::InputFloat("Training/Test Split", &test_split, 0.05f, 0.1f, "%.2f", NULL);
			ImGui::InputFloat("C", &gainC, 0.5f, 10.0f, "%.2f", NULL);
			ImGui::InputInt("Iterations", &num_iterations, 1, 10, NULL);
			
			if(ImGui::Button("Train & Test")) {
				test_completed = false;
				LARGE_INTEGER start;
				QueryPerformanceCounter(&start);
				
				DataSet dataset = ExtractLabelsAndGenerateMatrix(parsed_file, current_col);
				
				DataSet datasets[2];
				RandomSplitSample(0.6f,dataset, datasets);
				
				model = SVMTrain(datasets[0], gainC, .0001f, num_iterations);
				test_results = SVMTest(datasets[1], model);
				test_completed = true;
				LARGE_INTEGER endCounter;
				QueryPerformanceCounter(&endCounter);
				time_taken = ((f32)(endCounter.QuadPart - start.QuadPart) / (f32)g_PerformanceCountFreq);
			}
			
			if(test_completed) {
				ImGui::BeginChild("Scrolling");
				for(u32 i = 0; i < model->num_classifications; i++) {
					BinaryClassifier curr = model->classifiers[i];
					
					ImGui::Text("Model %d:  Labels: %s & %s  \n B Val: %.3f \n", i, curr.label_1, curr.label_2, curr.thresholdB);
				}
				
				ImGui::Text("Predictions:");
				// TODO(Cian): Write report and investigate the issue with classifications always(almost) being stout
				
				for(u32 i = 0; i < test_results->num_predicted; i++) {
					ImGui::Text("%s", test_results->predicted_labels[i]);
				}
				ImGui::Text("Took: %.2f seconds", time_taken);
				ImGui::Text("Accuracy: %.2f", test_results->accuracy);
				
				ImGui::EndChild();
			}
		}
		
		
		
		ImGui::End();
		ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		
		processInput(window);
		glfwSwapBuffers(window);
		glfwPollEvents();    
	}
	
	
	ImGui::DestroyContext();
	glfwTerminate();
	
	Memory_ArenaRelease(&g_MemoryInfo.permanent_arena);
	Memory_ArenaRelease(&g_MemoryInfo.file_arena);
	Memory_ArenaRelease(&g_MemoryInfo.temp_calc_arena);
	Memory_ArenaRelease(&g_MemoryInfo.calc_arena);
	
	return 0;
}

