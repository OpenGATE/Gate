/*

  This file is part of liblongvol.

  liblongvol is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  liblongvol is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with liblongvol; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/
#include "GateDMaplongvol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int Longvol::readLongvolData( FILE *fin ) {

  // Read header
  // Buf for a line
  char buf[128];
  int linecount = 1;
  int fieldcount = 0;

  // Read the file line by line until ".\n" is found
  for (	char *line = fgets( buf, 128, fin );
        line && strcmp( line, ".\n" ) != 0 ;
        line = fgets( line, 128, fin ), ++linecount
        ) {

    if (line[strlen(line) - 1] != '\n') {
      fprintf(debugFile, "LIBLONGVOL : Line %d too long\n", linecount );
      return 1;
    }

    int i;
    for (i = 0; (line[i] && (line[i] != ':')); ++i) {;}

    if (i == 0 || i >= 126 || line[i] != ':') {
      fprintf( debugFile, "LIBLONGVOL : Invalid header read at line %d\n", linecount );
      return 1;
    } else {

      if (fieldcount == MAX_HEADERNUMLINES) {
        fprintf( debugFile, "LIBLONGVOL : Too many lines in HEADER, ignoring\n" );
        continue;
      }
      if (fieldcount > MAX_HEADERNUMLINES)
        continue;

      // Remove \n from end of line
      if (line[ strlen(line) - 1 ] == '\n')
        line[ strlen(line) - 1 ] = 0;

      // hack : split line in two str ...
      line[i] = 0;
      header[ fieldcount++ ] = HeaderField( line, line + i + 2 );
      // +2 cause we skip the space
      // following the colon
    }

  }

  // Check required headers
  for (int i = 0; requiredHeaders[i]; ++i) {
    if (getHeaderValue( "Version" ) != NULL && (strcmp( requiredHeaders[i], "Int-Endian" ) == 0 || strcmp( requiredHeaders[i], "Lvoxel-Endian" ) == 0)) {
      continue;
    }
    if (getHeaderField( requiredHeaders[i] ) == -1) {
      fprintf( debugFile, "LIBLONGVOL : Required Header Field missing : %s\n", requiredHeaders[i] );
      return 1;
    }
  }
  // Check endian
  if (getHeaderValue( "Version" ) == NULL &&
      (strcmp( endian.i_endian.ci, getHeaderValue( "Int-Endian" )) != 0 ||
       strcmp( endian.v_endian.cv, getHeaderValue( "Lvoxel-Endian" )) != 0)) {

    fprintf( debugFile, "LIBLONGVOL : This file has incompatible endianess (%s <-> %s). Convertion to be implemented !\n", endian.i_endian.ci, getHeaderValue( "Int-Endian" ));

    if (strlen( endian.i_endian.ci ) == strlen( getHeaderValue( "Int-Endian" ) ) &&
        strlen( endian.v_endian.cv ) == strlen( getHeaderValue( "Lvoxel-Endian" ) ) ) {
      fprintf( debugFile, "LIBLONGVOL: Trying to continue...\n" );
    } else {
      fprintf( debugFile, "LIBLONGVOL: Abort...\n" );
    }

    return 1;

  }

  getHeaderValueAsInt("X", &sx);
  getHeaderValueAsInt("Y", &sy);
  getHeaderValueAsInt("Z", &sz);

  if (getHeaderValue( "Version" ) != NULL ) { // Field Version appeared only in v2
    return readV2RawData( fin, true, sx, sy, sz, alpha() );
  }
  return readV1RawData( fin, true, alpha() );

}


int Longvol::readV1RawData( FILE *fin, bool headerInited, lvoxel defaultAlpha ) {

  int count = 0;
  int rawsx, rawsy, rawsz;

  // Size of the longvolume
  count += fread( &rawsx, sizeof(int), 1, fin );
  count += fread( &rawsy, sizeof(int), 1, fin );
  count += fread( &rawsz, sizeof(int), 1, fin );

  if (count != 3) {
    fprintf( debugFile, "LIBLONGVOL : can't read file (raw header)\n" );
    return 1;
  }

  if (headerInited) {

    // The raw header contains the longvolume size too

    getHeaderValueAsInt("X", &sx);
    getHeaderValueAsInt("Y", &sy);
    getHeaderValueAsInt("Z", &sz);

    if (sx != rawsx || sy != rawsy || sz != rawsz) {
      fprintf( debugFile, "LIBLONGVOL : Warning : Incoherent longvol header with raw header !\n" );
    }

    int voxsize;
    if (getHeaderValueAsInt( "Lvoxel-Size", &voxsize ) == 0 && voxsize != sizeof(lvoxel)) {
      fprintf( debugFile, "LIBLONGVOL : This file was generated with a lvoxel-size that we do not support.\n");
      return 1;
    }

  }

  // We should have a useless \n in the file at this point
  char tmp;
  count = fread( &tmp, sizeof(char), 1, fin );

  if (count != 1 || tmp != '\n') {
    fprintf( debugFile, "LIBLONGVOL : I thouhgt I would have read a \\n !\n" );
    return 1;
  }

  return readV2RawData( fin, headerInited, sx, sy, sz, defaultAlpha );
}

int Longvol::readV2RawData( FILE *fin, bool headerInited, int sizeX, int sizeY, int sizeZ, lvoxel defaultAlpha ) {

  long count = 0;

  // now read the raw data
  sx = sizeX;
  sy = sizeY;
  sz = sizeZ;

  total = sx * sy * sz;
  try {
    data = new lvoxel[ sx * sy * sz ];
  } catch (...) {
    fprintf( debugFile, "LIBLONGVOL : not enough memory\n" );
    return 1;
  }

  count = fread( data, sizeof(lvoxel), total, fin );

  if (count != total) {
    fprintf( debugFile, "LIBLONGVOL : can't read file (raw data) !\n" );
    return 1;
  }

  // Now do some initializations
  // (if a header is not found, the second param is not changed)
  getHeaderValueAsInt( "Center-X", &cx );
  getHeaderValueAsInt( "Center-Y", &cy );
  getHeaderValueAsInt( "Center-Z", &cz );

  // If header has not been inited, do it !
  if (!headerInited) {
    setHeaderValue( "X", sx );
    setHeaderValue( "Y", sy );
    setHeaderValue( "Z", sz );
    setHeaderValue( "Alpha-Color", defaultAlpha );
    setHeaderValue( "Lvoxel-Size", (int)sizeof(lvoxel) );
    setHeaderValue( "Int-Endian", endian.i_endian.ci );
    setHeaderValue( "Lvoxel-Endian", endian.v_endian.cv );
    setHeaderValue( "Version", "2" );
  }

  return 0;

}

const char *Longvol::requiredHeaders[] = {
  "X", "Y", "Z", "Lvoxel-Size", "Int-Endian", "Lvoxel-Endian", "Alpha-Color", NULL
};
