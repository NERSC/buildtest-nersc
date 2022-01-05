
int find_gpus(void);
void gpu_pci_id(char* device_id, int device_num);
void set_my_device(int my_device);
int get_current_device();
void vec_add_gpu(double *h_a, double *h_b, double *h_c, int n);
