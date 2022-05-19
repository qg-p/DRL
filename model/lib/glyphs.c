void translate_glyphs(void*DST, void*SRC, void*GLYPH_ARRAY) {
    short(*glyph_array)[0][4]         = GLYPH_ARRAY;
    short(*dst)           [4][21][79] = DST;
    short(*src)              [21][79] = SRC;
    short*srcrow;
    short*dstrow[4];
    for (int row=0;row<21;row++) {
        srcrow = (*src)[row];
        dstrow[0] = (*dst)[0][row];
        dstrow[1] = (*dst)[1][row];
        dstrow[2] = (*dst)[2][row];
        dstrow[3] = (*dst)[3][row];
        for (int col=0;col<79;col++) {
            short glyph = *srcrow++;
            short*array = (*glyph_array)[glyph];
            *dstrow[0]++ = *array++;
            *dstrow[1]++ = *array++;
            *dstrow[2]++ = *array++;
            *dstrow[3]++ = *array++;
        }
    }
}