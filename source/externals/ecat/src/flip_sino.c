flip_sino() {
short *data1, *plane0, *plane1, *line0, *line1;

/* dimensions : if storage_order = 0  i.e.
     (((projs x z_elements)) x num_angles) x Ringdiffs
    read and flip y and z
*/
       if (scan3Dsub->storage_order == 0) {
            sx = data->xdim; sz = data->ydim; sy = data->zdim;
            data1 = (short*)data->data_ptr;
            file_pos = (matdir.strtblk+1)*MatBLKSIZE;
            slice_blks = (sx*sy*sizeof(short)+511)/512;
            plane0 = (short*)malloc(512*slice_blks);
            for (z=0; z<sz; z++) {      /* for each planar view fixed theta */
                fseek(mptr->fptr,file_pos,0);
                fread(plane0,sizeof(short),sx*sy,mptr->fptr);
                file_data_to_host(plane0,slice_blks,scan3Dsub->data_type);
                file_pos += sx*sy*sizeof(short);
                for (y=0; y<sy; y++) {      /* for each line fixed plane */
                    line0 =  plane0 + (sx*y);
                    plane1 = data1 + (sx*sz*y);
                    line1 = plane1 + (sx*z);
                    memcpy(line1,line0,sx*sizeof(short));
                }
            }
            scan3Dsub->storage_order = 1;
            free(plane0);

flip_attn() {
  float *fdata1, *fplane0, *fplane1, *fline0, *fline1;
        if (attnsub->storage_order == 0) { /*  read and flip y and z */
            sx = data->xdim; sz = data->ydim; sy = data->zdim;
            fdata1 = (float*)data->data_ptr;
            file_pos = matdir.strtblk*MatBLKSIZE;
            slice_blks = (sx*sy*sizeof(float)+511)/512;
            fplane0 = (float*)malloc(512*slice_blks);
            for (z=0; z<sz; z++) {      /* for each planar view fixed theta */
                fseek(mptr->fptr,file_pos,0);
                fread(fplane0,sizeof(float),sx*sy,mptr->fptr);
                file_data_to_host(fplane0,slice_blks,attnsub->data_type);
                file_pos += sx*sy*sizeof(float);
                for (y=0; y<sy; y++) {      /* for each line fixed plane */
                    fline0 =  fplane0 + (sx*y);
                    fplane1 = fdata1 + (sx*sz*y);
                    fline1 = fplane1 + (sx*z);
                    memcpy(fline1,fline0,sx*sizeof(float));
                }
            }
            attnsub->storage_order = 1;
            free(fplane0);
}
