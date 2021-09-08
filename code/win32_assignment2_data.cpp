#define HASH_SIZE 256
global MemoryArena* g_Current_Arena;

#define PUSH_ARENA(m) g_Current_Arena = &m

struct FileRead {
	char *file_start;
	u64 file_size;
	b32 success;
};

struct FileLine {
	char **entries;
	FileLine *next;
};

struct ParsedFile {
	u32 num_cols;
	u32 num_lines;
	FileLine *header;
	FileLine *lines;
};

/*
Matrix ordered as follows (row-ordered):

  0 1 2
3 4 5
6 7 8
*/

struct Matrix {
	u32 cols, rows;
	f32 *data;
};

struct DataSet {
	Matrix features;
	char **label_vector;
	u32 label_size;
};

struct BinaryClassifier {
	Matrix lagrangian_multipliers;
	Matrix features;
	Matrix labels;
	Matrix w_vector;
	f32 thresholdB;
	// NOTE(Cian): label_1 is always := +1 label_2 is always := -1
	char *label_1;
	char *label_2;
};

struct SVMModel {
	BinaryClassifier *classifiers;
	u32 num_classifications;
	char **unique_label_list;
	u32 unique_label_size;
};

struct HashEntry {
	u32 count;
	char *label;
};

struct SVMTestResults {
	f32 accuracy;
	char **predicted_labels;
	u32 num_predicted;
};
internal FileRead  Win32_ReadEntireFile(char *file_path) {
	FileRead result = {};
	result.success = false;
	
	HANDLE file_handle = CreateFileA((LPCSTR)file_path, GENERIC_READ, NULL, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if(file_handle != INVALID_HANDLE_VALUE) {
		LARGE_INTEGER file_size_struct = {};
		
		if(GetFileSizeEx(file_handle, &file_size_struct)) {
			u64 file_size = (u64)file_size_struct.QuadPart + 1;
			result.file_size = file_size;
			
			void *file_buffer = Memory_ArenaPush(&g_MemoryInfo.file_arena, file_size);
			result.file_start = (char *)file_buffer;
			
			u32 bytes_read = 0;
			if(ReadFile(file_handle, file_buffer, (DWORD)file_size, (LPDWORD)&bytes_read, NULL)) {
				result.success = true;
				result.file_start[result.file_size] = '\0';
			}
		}
		CloseHandle(file_handle);
	}
	
	return result;
} 

internal void StringCopy(char *destination, char *source, u32 length) {
	for(u32 i = 0; i < length; i++) {
		destination[i] = source[i];
	}
}

internal u32 GetNumCols(char *line) {
	u32 num_cols = 0;
	char *current_char = line;
	while(*current_char != '\n') {
		if(*current_char == '\t') {
			num_cols++;
		} 
		current_char++;
	}
	
	return num_cols + 1;
}

internal void SaveToken(FileLine *current, char *string_buffer, u32 buffer_idx, u32 token_index) {
	string_buffer[buffer_idx] = '\0';
	
	current->entries[token_index] = (char*)Memory_ArenaPush(&g_MemoryInfo.permanent_arena, sizeof(char) * (buffer_idx + 1));
	
	StringCopy(current->entries[token_index], string_buffer, buffer_idx + 1);
}

internal ParsedFile ParseFile(FileRead file_read) {
	ParsedFile result = {};
	result.num_cols = GetNumCols(file_read.file_start);
	
	char string_buffer[2048];
	u32 curr_buffer_index = 0;
	u32 token_index = 0;
	u32 num_lines = 0;
	
	FileLine *curr = (FileLine*)Memory_ArenaPush(&g_MemoryInfo.file_arena, sizeof(FileLine));
	curr->entries = (char**)Memory_ArenaPush(&g_MemoryInfo.file_arena, sizeof(char*) * result.num_cols);
	result.lines = curr;
	
	FileLine *header = (FileLine*)Memory_ArenaPush(&g_MemoryInfo.file_arena, sizeof(FileLine));
	header->entries = (char**)Memory_ArenaPush(&g_MemoryInfo.file_arena, sizeof(char*) * result.num_cols);
	result.header = header;
	
	for(u64 bytes_read = 0; bytes_read < file_read.file_size; bytes_read++) {
		char current = file_read.file_start[bytes_read];
		
		if(current == '\r') {
			if(num_lines == 0) {
				SaveToken(header, string_buffer, curr_buffer_index, token_index);
			} else {
				SaveToken(curr, string_buffer, curr_buffer_index, token_index);
				FileLine *new_line = (FileLine*)Memory_ArenaPush(&g_MemoryInfo.file_arena, sizeof(FileLine));
				new_line->entries = (char**)Memory_ArenaPush(&g_MemoryInfo.file_arena, sizeof(char*) * result.num_cols);
				curr->next = new_line;
				curr = new_line; 
			}
			token_index = 0;
			curr_buffer_index = 0;
			num_lines++;
			// NOTE(Cian): to prevent \r getting consumed in string
			bytes_read++;
			
		} else if(current == '\0') {
			SaveToken(curr, string_buffer, curr_buffer_index, token_index);
			num_lines++;
		} else if(current == '\t') {
			if(num_lines == 0)
				SaveToken(header, string_buffer, curr_buffer_index, token_index);
			else
				SaveToken(curr, string_buffer, curr_buffer_index, token_index);
			
			curr_buffer_index = 0;
			token_index++;
		} else {
			string_buffer[curr_buffer_index] = current;
			curr_buffer_index++;
		}
	}
	result.num_lines = num_lines - 1;
	return result;
}

inline Matrix init_matrix(u32 rows, u32 cols){
	Matrix result = {};
	result.rows = rows;
	result.cols = cols;
	result.data = (f32*)Memory_ArenaPush(g_Current_Arena,sizeof(f32) * (cols * rows));
	
	return result;
}

inline void set_matrix_value(Matrix *matrix, u32 row, u32 col, f32 value) {
	u32 index = (row * matrix->cols) + col;
	matrix->data[index] = value;
}

inline f32 get_matrix_value(Matrix *matrix, u32 row, u32 col) {
	u32 index = (row * matrix->cols) + col;
	return matrix->data[index];
}

internal Matrix transpose_matrix(Matrix original) {
	Matrix transpose = init_matrix(original.cols, original.rows);
	
	for(u32 i = 0; i < transpose.rows; i++) {
		for(u32 j = 0; j < transpose.cols; j++) {
			set_matrix_value(&transpose, i, j, get_matrix_value(&original,j, i));
		}
	}
	return transpose;
}

internal Matrix multiply_matrix(Matrix a, Matrix b) {
	assert(a.cols == b.rows);
	
	Matrix result = init_matrix(a.rows, b.cols);
	
	for(u32 i = 0; i < result.rows; i++) {
		for(u32 j = 0; j < result.cols; j++) {
			set_matrix_value(&result, i, j, 0.0f);
			for(u32 k = 0; k < a.cols; k++) {
				// NOTE(Cian): To prevent float underflow issue
				f32 result_val = get_matrix_value(&result, i,j);
				f32 a_val = get_matrix_value(&a, i, k);
				f32 b_val = get_matrix_value(&b, k, j);
				f32 val = result_val + (a_val * b_val);
				if(val == 0) val = 0;
				
				set_matrix_value(&result, i, j, val);
			}
		}
	}
	
	return result;
}

internal Matrix multiply_elements_matrix(Matrix a, Matrix b) {
	assert(a.cols == b.cols && a.rows == b.rows);
	
	Matrix result = init_matrix(a.rows, a.cols);
	
	
	for(u32 i = 0; i < result.rows; i++) {
		for(u32 j = 0; j < result.cols; j++) {
			f32 a_val = get_matrix_value(&a, i, j);
			f32 b_val = get_matrix_value(&b, i, j);
			f32 val = a_val * b_val;
			// NOTE(Cian): To prevent float underflow that we were experiencing
			if(val == 0) val = 0;
			set_matrix_value(&result, i, j, val);
		}
	}
	
	return result;
}

internal Matrix multiply_scalar_matrix(Matrix a, f32 scalar) {
	Matrix result = init_matrix(a.rows, a.cols);
	
	for(u32 i = 0; i < result.rows; i++) {
		for(u32 j = 0; j < result.cols; j++) {
			f32 val = get_matrix_value(&result, i, j) * scalar;
			// NOTE(Cian): To prevent float underflow that we were experiencing
			if(val == 0) val = 0;
			set_matrix_value(&result, i, j, val);
		}
	}
	
	return result;
}

internal f32 sum_matrix(Matrix a) {
	f32 result = 0.0f;
	for(u32 i = 0; i < a.rows; i++) {
		for(u32 j = 0; j < a.cols; j++) {
			result += get_matrix_value(&a, i, j);
		}
	}
	
	return result;
}

internal Matrix extract_row(Matrix a, u32 row) {
	Matrix result = init_matrix(1, a.cols);
	
	for(u32 col = 0; col < a.cols; col++) {
		set_matrix_value(&result, 0, col, get_matrix_value(&a, row, col));
	}
	
	return result;
}

internal void print_matrix(Matrix a) {
	for(u32 i = 0; i < a.rows; i++) {
		for(u32 j = 0; j < a.cols; j++) {
			f32 val = get_matrix_value(&a, i, j);   
			if((j+1) == a.cols)
				printf("%.20f \n", val);
			else
				printf("%.20f ",val);
		}
	}
}
internal void print_matrices(Matrix a, Matrix b) {
	b = transpose_matrix(b);
	for(u32 i = 0; i < a.rows; i++) {
		for(u32 j = 0; j < a.cols; j++) {
			f32 val = get_matrix_value(&a, i, j);   
			printf("%.2f ",val);
		}
		f32 bval = get_matrix_value(&b, i, 0);
		printf(" %.2f \n", bval);
	}
}

internal void print_dataset(DataSet set) {
	for(u32 i = 0; i < set.features.rows; i++) {
		for(u32 j = 0; j < set.features.cols; j++) {
			f32 val = get_matrix_value(&set.features, i, j);   
			printf("%.2f ",val);
		}
		printf(" %s \n", set.label_vector[i]);
	}
}

internal DataSet ExtractLabelsAndGenerateMatrix(ParsedFile file, u32 label_column) {
	DataSet dataset = {};
	dataset.label_vector = (char**)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(char*) * file.num_lines);
	dataset.label_size = file.num_lines;
	
	PUSH_ARENA(g_MemoryInfo.calc_arena);
	Matrix matrix = init_matrix(file.num_lines, file.num_cols - 1);
	// NOTE(Cian): Loop through the files lines, add features to matrix, and labels to the label_vector
	FileLine *current = file.lines;
	for(u32 row = 0; row < file.num_lines; row++) {
		
		for(u32 col = 0; col < file.num_cols; col++) {
			char *entry = current->entries[col];
			
			if(col == label_column) {
				dataset.label_vector[row] = entry;
			} else {
				// NOTE(Cian): Parse string to f32
				// NOTE(Cian): Not checking if string is a valid float right now
				u32 m_col = col >= label_column ? col - 1: col;
				set_matrix_value(&matrix, row, m_col, strtof(entry, NULL));
			}
		}
		current = current->next;
	}
	dataset.features = matrix;
	return dataset;
}

// NOTE(Cian): Very basic rounding
inline u32 BasicFloatToIntRound(f32 num) {
	return (u32)(num + 0.5);
}

internal u64 RandomMax(u64 max) {
	u64 num_bins = max + 1;
	u64 num_rand = RAND_MAX + 1;
	u64 bin_size = num_rand / num_bins;
	u64 defect = num_rand % num_bins;
	
	u64 x = 0;
	do {
		x = rand();
	} while(num_rand - defect <= x);
	
	return x/bin_size;
}

// NOTE(Cian): training_split is in range 0.0f -> 1.0f
internal void RandomSplitSample(f32 training_split, DataSet source, DataSet *sets) {
	
	PUSH_ARENA(g_MemoryInfo.calc_arena);
	u32 num_total_rows = source.features.rows;
	u32 num_total_cols = source.features.cols;
	
	u32 num_training = BasicFloatToIntRound((f32)num_total_rows * training_split);
	u32 num_testing = BasicFloatToIntRound((f32)num_total_rows * (1.0f - training_split));
	
	b32 *taken_flags = (b32*)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(b32) * num_total_rows);
	
	DataSet training = {};
	training.features = init_matrix(num_training, num_total_cols);
	training.label_vector = (char**)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(char*) * num_training);
	training.label_size = num_training;
	DataSet testing = {};
	testing.features = init_matrix(num_testing, num_total_cols);
	testing.label_vector = (char**)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(char*) * num_testing);
	testing.label_size = num_testing;
	
	// NOTE(Cian): Loop through training set and insert values
	for(u32 row = 0; row < num_training; row++) {
		u32 idx_to_take = (u32)RandomMax(num_total_rows - 1);
		
		while(taken_flags[idx_to_take]) {
			idx_to_take = (u32)RandomMax(num_total_rows - 1);
		}
		taken_flags[idx_to_take] = true;
		training.label_vector[row] = source.label_vector[idx_to_take];
		for(u32 col = 0; col < num_total_cols; col++) {
			set_matrix_value(&training.features, row, col, get_matrix_value(&source.features, idx_to_take, col));
		}
	}
	
	for(u32 row = 0; row < num_testing; row++) {
		u32 idx_to_take = (u32)RandomMax(num_total_rows - 1);
		
		while(taken_flags[idx_to_take]) {
			idx_to_take = (u32)RandomMax(num_total_rows - 1);
		}
		taken_flags[idx_to_take] = true;
		testing.label_vector[row] = source.label_vector[idx_to_take];
		for(u32 col = 0; col < num_total_cols; col++) {
			set_matrix_value(&testing.features, row, col, get_matrix_value(&source.features, idx_to_take, col));
		}
	}
	sets[0] = training;
	sets[1] = testing;
	
}

internal b32 HasStringAlreadyAppeared(char *key, char **hash_list) {
	u32 hash_value = StringToCRC32(key);
	u32 hash_index = hash_value & (HASH_SIZE - 1);
	
	char *curr = hash_list[hash_index];
	
	while(curr) {
		if(strcmp(curr, key) == 0) {
			return true;
		}
		hash_index = (hash_index + 1) & (HASH_SIZE - 1);
		curr = hash_list[hash_index];
	}
	
	if(curr == NULL) {
		hash_list[hash_index] = key;
		return false;
	} 
	
	return false;
}


internal void FilterMatrix(DataSet training, char *label_1, char *label_2, Matrix *new_matrix, Matrix *new_vector) {
	// NOTE(Cian): Going to set each Matrix to be the same size as full set initially, then
	// change it's row value at end, will waste some memory but it's ok for our purposes
	
	PUSH_ARENA(g_MemoryInfo.calc_arena);
	(*new_matrix) = init_matrix(training.features.rows, training.features.cols); 
	(*new_vector) = init_matrix(1, training.features.rows); 
	u32 curr_row = 0;
	for(u32 i = 0; i < training.label_size; i++) {
		// NOTE(Cian): label_1 will always be +1, label_2 will always be -1
		if(strcmp(label_1, training.label_vector[i]) == 0) {
			for(u32 j = 0; j < training.features.cols; j++) {
				set_matrix_value(new_matrix, curr_row, j, get_matrix_value(&training.features, i, j));
				set_matrix_value(new_vector, 0, curr_row, 1.0f);
			}
			curr_row++;
		} else if(strcmp(label_2, training.label_vector[i]) == 0) {
			for(u32 j = 0; j < training.features.cols; j++) {
				set_matrix_value(new_matrix, curr_row, j, get_matrix_value(&training.features, i, j));
				set_matrix_value(new_vector, 0, curr_row, -1.0f);
			}
			curr_row++;
		}
	}
	new_matrix->rows = curr_row;
	new_vector->cols = curr_row;
}

internal u32 RandomValueNotEqualTo(u32 i, u32 m) {
	u32 j = i;
	
	while(j == i) {
		j = (u32)RandomMax(m);
	}
	
	return j;
}

internal f32 Max(f32 a, f32 b) {
	if(a > b) {
		return a;
	} else {
		return b;
	}
}

internal f32 Min(f32 a, f32 b) {
	if(a < b) {
		return a;
	} else {
		return b;
	}
}

inline f32 Abs(f32 a) {
	f32 result = a;
	if(result < 0) {
		result = -result;
	}
	
	return result;
}

internal void SimpleSMO(Matrix features, Matrix labels , f32 C, f32 tol, u32 max_samples, BinaryClassifier *classifier) {
	u32 m = features.rows;
	f32 b = 0.0f;
	
	// NOTE(Cian): Our Langrangian Multipliers, init_matrix sets each value to 0 by default since our MemoryArenas are zero initialised
	PUSH_ARENA(g_MemoryInfo.calc_arena);
	Matrix alphas = init_matrix(m,1);
	u32 iteration = 0;
	
	/*printf("Classification %s & %s \n", classifier->label_1, classifier->label_2);
	print_matrices(features, labels);*/
	
	while(iteration < max_samples) {
		u32 num_changed_alphas = 0;
		
		for(u32 i = 0; i < m; i++) {
			// NOTE(Cian): Calculate Ei = f(xi) - y(i) where f(xi) = (alphai*yi) * (xi * x) + b
			PUSH_ARENA(g_MemoryInfo.temp_calc_arena);
			Matrix alphai_yi = transpose_matrix(multiply_elements_matrix(alphas, transpose_matrix(labels)));
			Matrix data_i = multiply_matrix(features, transpose_matrix(extract_row(features, i)));
			f32 fxi = sum_matrix(multiply_matrix(alphai_yi, data_i)) + b;
			f32 Ei = fxi - get_matrix_value(&labels, 0, i);
			if((get_matrix_value(&labels, 0, i) * Ei < -tol && get_matrix_value(&alphas, i, 0) < C) || (get_matrix_value(&labels, 0, i) * Ei > tol && get_matrix_value(&alphas, i, 0) > 0)) {
				
				u32 j = RandomValueNotEqualTo(i, m);
				
				Matrix alphaj_yj = transpose_matrix(multiply_elements_matrix(alphas, transpose_matrix(labels)));
				Matrix data_j = multiply_matrix(features, transpose_matrix(extract_row(features, j)));
				f32 fxj = sum_matrix(multiply_matrix(alphaj_yj, data_j)) + b;
				f32 Ej = fxj - get_matrix_value(&labels, 0, j);
				
				f32 old_alphai = get_matrix_value(&alphas, i, 0);
				f32 old_alphaj = get_matrix_value(&alphas, j, 0);
				
				f32 L = 0;
				f32 H = 0;
				if(get_matrix_value(&labels, 0, i) != get_matrix_value(&labels, 0, j)) {
					L = Max(0, get_matrix_value(&alphas, j, 0) - get_matrix_value(&alphas, i, 0));
					H = Min(C, C + get_matrix_value(&alphas, j, 0) - get_matrix_value(&alphas, i, 0));
				} else {
					L = Max(0, get_matrix_value(&alphas, j, 0) + get_matrix_value(&alphas, i, 0) - C);
					H = Min(C, get_matrix_value(&alphas, j, 0) + get_matrix_value(&alphas, i, 0));
				}
				
				if(L == H) continue;
				
				Matrix featuresRowI = extract_row(features, i);
				Matrix featuresRowJ = extract_row(features, j);
				Matrix lhsETA = multiply_scalar_matrix(multiply_matrix(featuresRowI,transpose_matrix(featuresRowJ)), 2.0f);
				Matrix midETA = multiply_matrix(featuresRowI, transpose_matrix(featuresRowI));
				Matrix rhsETA = transpose_matrix(multiply_matrix(featuresRowJ, transpose_matrix(featuresRowJ))); 
				
				f32 eta = sum_matrix(lhsETA) - sum_matrix(midETA) - sum_matrix(rhsETA);
				
				if(eta >= 0) continue;
				
				f32 labelj = get_matrix_value(&labels, 0, j);
				f32 alphaj = get_matrix_value(&alphas, j, 0);
				f32 new_alphaj = alphaj - ((labelj * (Ei-Ej)) / eta);
				
				if(new_alphaj > H)
					new_alphaj = H;
				if(new_alphaj < L) 
					new_alphaj = L;
				
				set_matrix_value(&alphas, j, 0, new_alphaj);
				
				if(Abs(get_matrix_value(&alphas, j, 0) - old_alphaj) < 0.00001f) continue;
				f32 alphai = get_matrix_value(&alphas, i, 0);
				f32 labeli = get_matrix_value(&labels, 0, i);
				f32 new_alphai = alphai + (labelj * labeli * (old_alphaj - new_alphaj));
				
				
				set_matrix_value(&alphas, i, 0, new_alphai);
				
				f32 delta_alphai = new_alphai - old_alphai;
				f32 delta_alphaj = new_alphaj - old_alphaj;
				
				f32 left_scalar = labeli * delta_alphai;
				f32 right_scalar = labelj * delta_alphaj;
				
				// NOTE(Cian): These will result in 1x1 matrices
				Matrix left_mult_matrix = multiply_matrix(featuresRowI, transpose_matrix(featuresRowI));
				Matrix right_mult_matrix = multiply_matrix(featuresRowI, transpose_matrix(featuresRowJ));
				
				f32 b1 = b - Ej - (left_scalar * sum_matrix(left_mult_matrix)) - (right_scalar * sum_matrix(right_mult_matrix));
				
				left_mult_matrix = multiply_matrix(featuresRowI, transpose_matrix(featuresRowJ));
				right_mult_matrix = multiply_matrix(featuresRowJ, transpose_matrix(featuresRowJ));
				
				f32 b2 =  b - Ej - (left_scalar * sum_matrix(left_mult_matrix)) - (right_scalar * sum_matrix(right_mult_matrix));
				
				if(0 < new_alphai && C > new_alphai)
					b = b1;
				else if(0 < new_alphaj && C > new_alphaj)
					b = b2;
				else
					b = (b1 + b2) / 2.0f;
				
				num_changed_alphas++;
			}
		}
		Memory_ArenaClear(&g_MemoryInfo.temp_calc_arena);
		if(num_changed_alphas == 0)
			iteration++;
		else
			iteration = 0;
	}
	
	classifier->lagrangian_multipliers = alphas;
	classifier->thresholdB = b;
}

// NOTE(Cian): Using a simplified SMO algorithm, potential for it not to resolve if a 
// linear line cannot seperate the data, further work would implement the "kernel trick" e.g.
// artifically increasing the dimensionality 
internal SVMModel* SVMTrain(DataSet training, f32 hingeC, f32 tol, u32 max_samples) {
	SVMModel *result = (SVMModel*)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(SVMModel));
	
	Matrix *feature_matrix = &training.features;
	u32 unique_class_count = 0;
	
	char *hash_list[HASH_SIZE] = {0};
	for(u32 i = 0; i < training.label_size; i++) {
		char *curr_label = training.label_vector[i];
		
		if(HasStringAlreadyAppeared(curr_label, hash_list) == false) {
			unique_class_count++;
		}
	}
	
	u32 num_binary_comparisons = (unique_class_count * (unique_class_count -1)) / 2;
	char **unique_label_list = (char**)Memory_ArenaPush(&g_MemoryInfo.permanent_arena, sizeof(char*) * unique_class_count);
	// NOTE(Cian): Little piggy again, looping through the entire hash_list.
	u32 label_idx = 0;
	for(u32 i = 0; i < HASH_SIZE; i++) {
		char *curr = hash_list[i];
		
		if(curr) {
			unique_label_list[label_idx] = curr;
			label_idx++;
		}
	}
	
	u32 num_classifications_per_class = (num_binary_comparisons * 2) / unique_class_count;
	u32 *classifications_per_class = (u32*)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(u32) * unique_class_count);
	
	Matrix *filtered_feature_matrices = (Matrix*)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(Matrix) * num_binary_comparisons);
	Matrix *filtered_label_vectors = (Matrix*)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(Matrix) * num_binary_comparisons);
	
	
	BinaryClassifier *classifiers = (BinaryClassifier*)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(BinaryClassifier) * num_binary_comparisons);
	
	
	u32 label_1 = 0;
	u32 label_2 = 1;
	for(u32 i = 0; i < num_binary_comparisons; i++) {
		if(classifications_per_class[label_1] >= num_classifications_per_class) {
			label_1++;
		}
		
		if(label_1 != label_2) {
			if(classifications_per_class[label_2] < num_classifications_per_class) {
				FilterMatrix(training,unique_label_list[label_1], unique_label_list[label_2], &filtered_feature_matrices[i], &filtered_label_vectors[i]);
				
				classifiers[i].label_1 = unique_label_list[label_1];
				classifiers[i].label_2 = unique_label_list[label_2];
				classifiers[i].features = filtered_feature_matrices[i];
				classifiers[i].labels = filtered_label_vectors[i];
				
				SimpleSMO(filtered_feature_matrices[i],filtered_label_vectors[i] , hingeC, tol, max_samples, classifiers + i);
				
				classifications_per_class[label_1]++;
				classifications_per_class[label_2]++;
			}
		}
		
		if(label_2 >= unique_class_count - 1) 
			label_2 = label_1 + 2;
		else
			label_2++;
	}
	
	result->classifiers = classifiers;
	result->num_classifications = num_binary_comparisons;
	result->unique_label_list = unique_label_list;
	result-> unique_label_size = unique_class_count;
	
	return result;
}

internal SVMTestResults* SVMTest(DataSet dataset, SVMModel *model) {
	PUSH_ARENA(g_MemoryInfo.temp_calc_arena);
	SVMTestResults *result = (SVMTestResults*)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(SVMTestResults));
	
	result->predicted_labels = (char**)Memory_ArenaPush(&g_MemoryInfo.calc_arena, sizeof(char*) * dataset.label_size);
	result->num_predicted = dataset.label_size;
	
	u32 correct_classifications = 0;
	
	for(u32 i = 0; i < dataset.label_size; i++) {
		
		HashEntry label_votes[HASH_SIZE] = {0};
		Matrix row = extract_row(dataset.features, i);
		for(u32 j = 0; j < model->num_classifications; j++) {
			BinaryClassifier current = model->classifiers[j];
			char *label;
			
			//printf("\n W Vector \n");
			Matrix alphas_labels =  transpose_matrix(multiply_elements_matrix(current.lagrangian_multipliers, transpose_matrix(current.labels)));
			
			Matrix w = transpose_matrix(multiply_matrix(alphas_labels, current.features));
			
			f32 fx = sum_matrix(multiply_matrix(row, w)) + current.thresholdB;
			
			if(fx > 0){
				label = current.label_1;
			} else {
				label = current.label_2;
			}
			
			// NOTE(Cian): Add a "vote" for the winning label for this classifier
			u32 hash_value = StringToCRC32(label);
			u32 hash_index = hash_value & (HASH_SIZE - 1);
			
			HashEntry *curr = &label_votes[hash_index];
			
			while(curr->label) {
				if(strcmp(curr->label, label) == 0) {
					curr->count++;
					break;
				}
				hash_index = (hash_index + 1) & (HASH_SIZE - 1);
				curr = &label_votes[hash_index];
			}
			
			if(curr->label == NULL) {
				curr->count = 1;
				curr->label = label;
			}
		}
		// NOTE(Cian): Bit piggy again
		HashEntry *winner = &label_votes[0];
		for(u32 k = 1; k < HASH_SIZE; k++) {
			HashEntry *current = &label_votes[k];
			if(current->count > winner->count) {
				winner = current;
			}
		}
		
		result->predicted_labels[i] = winner->label;
		
		if(strcmp(winner->label, dataset.label_vector[i]) == 0) {
			correct_classifications++;
		}
	}
	Memory_ArenaClear(&g_MemoryInfo.temp_calc_arena);
	result->accuracy = ((f32)correct_classifications / (f32)dataset.label_size) * 100.0f;
	return result;
}