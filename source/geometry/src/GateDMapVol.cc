/*

This file is part of libvol.

    libvol is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    libvol is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libvol; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

*/
#include "GateDMapVol.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/types.h>
#include <fcntl.h>

#include <errno.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

inline int max( int a, int b ) {
	return a > b ? a : b;
}

static int invmat33(double inv[3][3], double mat[3][3]);

Vol::Vol( 	int ssx, int ssy, int ssz, voxel defaultcolor ) :
	sx( ssx ), sy( ssy ), sz( ssz ), total( ssx * ssy * ssz ), state_ok(true),
	cx( ssx/2 ), cy( ssy/2 ), cz( ssz/2 ) {

	try {
		// new throws on failure, but do NOT return null
		data = new voxel[ total ];

		// fill data with default color
		for (int i = 0; i < total; ++i)
			data[i] = defaultcolor;

		setHeaderValue( "X", sx );
		setHeaderValue( "Y", sy );
		setHeaderValue( "Z", sz );
		setHeaderValue( "Voxel-Size", (int)sizeof( voxel ) );
		setHeaderValue( "Alpha-Color", defaultcolor );
		setHeaderValue( "Int-Endian", endian.i_endian.ci );
		setHeaderValue( "Voxel-Endian", endian.v_endian.cv );
		setHeaderValue( "Version", "2" );

	} catch (...) {
		fprintf( debugFile, "LIBVOL : couldn't allocate %d KBytes !\n", total/1024 );
		state_ok = false;
	}

}

Vol::Vol( const char *filename ) {

	state_ok = true;
	data = NULL;

	FILE *		fin;
	int 		errcode;
	int 		axis = 0;

    cx=cy=cz=0;

	// If filename is "", we read from stdin
	if (strcmp(filename, "") == 0) {
		fin = stdin;
	} else {
		fin = fopen( filename, "r" );
	}

	if (fin == NULL) {
		fprintf( debugFile, "LIBVOL : libvol : can't open %s.\n", filename );
		state_ok = false;
		sx = sy = sz = cx = cy = cz = total = 0;
		return;
	}

	errcode = readVolData( fin );
    //default behavior :
    cx = sx / 2;
    cy = sy / 2;
    cz = sz / 2;

	if (errcode != 0) {
		// put the object into a good state
		delete []data;
		data = NULL;
		sx = sy = sz = cx = cy = cz = total = 0;
		state_ok = false;
		if (fin != stdin) fclose( fin );
		return;
	}

	getHeaderValueAsInt("Axis", &axis);
	if (axis) {
		drawAxis();
	}

	if (fin != stdin)
		fclose( fin );

}

Vol::Vol( const char *fname, int sizeX, int sizeY, int sizeZ, voxel defaultAlpha ) :
	cx( sizeX/2 ), cy( sizeY/2 ), cz( sizeZ/2 )
{

	FILE *fin;

	if (strcmp( fname, "" ) == 0)
		fin = stdin;
	else
		fin = fopen( fname, "r" );
	if (fin == NULL) {
		fprintf( debugFile, "LIBVOL : libvol : can't open %s.\n", fname );
		state_ok = false;
		sx = sy = sz = cx = cy = cz =  total = 0;
		return;
	}

	int errcode = readV2RawData( fin, false, sizeX, sizeY, sizeZ, defaultAlpha );

	if (errcode != 0) {
		delete []data;
		data = NULL;
		sx = sy = sz = cx = cy = cz = total = 0;
		state_ok = false;
		if (fin != stdin) fclose( fin );
		return;
	}

}

Vol::Vol( const Vol &v ) {

	copy( v );

}

Vol::Vol() :
	sx(0), sy(0), sz(0), total(0),
	cx(0), cy(0), cz(0) {

	state_ok = true;
	data = 0;

	setHeaderValue( "X", sx );
	setHeaderValue( "Y", sy );
	setHeaderValue( "Z", sz );
	setHeaderValue( "Voxel-Size", (int)sizeof( voxel ) );
	setHeaderValue( "Alpha-Color", 0 );
	setHeaderValue( "Int-Endian", endian.i_endian.ci );
	setHeaderValue( "Voxel-Endian", endian.v_endian.cv );


}

void Vol::copy( const Vol &v ) {

	if (!v.state_ok) {
		state_ok = false;
		return;
	}

	try {
		data = new voxel[ v.total ];
		memcpy( data, v.data, v.total );
	} catch (...) {
		fprintf( debugFile, "LIBVOL : Can not allocate %d KBytes !\n", v.total/1024 );
		state_ok = false;
	}

	int defaultcolor = 0;
	v.getHeaderValueAsInt( "Alpha-Color", &defaultcolor );

	sx = v.sx;
	sy = v.sy;
	sz = v.sz;

    //Center copy
	cx = v.cx;
	cy = v.cy;
	cz = v.cz;

	total = v.total;

	for (int i = 0; i < MAX_HEADERNUMLINES; ++i)
		header[i] = v.header[i];

	state_ok = true;
}

void Vol::destroy() {
	delete []data;
}

const Vol &Vol::operator = (const Vol &v) {
	destroy();
	copy( v );
	return *this;
}

Vol::~Vol() {
	destroy();
}

voxel & Vol::operator()(int x, int y, int z) {
	assert( state_ok );
	assert( inBounds(x, y, z) );
	// Check bounds in debug mode
	x -= minX();
	y -= minY();
	z -= minZ();
//	return data[ z*sy*sx + y*sx + x ];
	return data[ posOf( x, y, z ) ];
}


voxel Vol::operator()(int x, int y, int z) const {
	assert( state_ok );
	assert( inBounds(x, y, z) );
	// Check bounds in debug mode
	x -= minX();
	y -= minY();
	z -= minZ();
//	return data[ z*sy*sx + y*sx + x ];
	return data[ posOf( x, y, z ) ];
}
int Vol::dumpVol( const char *fname ) {

	assert( state_ok );

	FILE *f;

	if (strcmp( fname, "" ) == 0) {
		f = stdout;
	}
	else {
		f = fopen( fname, "w" );
		if (f == NULL) {
			fprintf( debugFile, "LIBVOL : can not open \"%s\" : %s\n", fname, strerror(errno) );
			return 1;
		}
	}

	// Write header
	for (int i = 0; i < MAX_HEADERNUMLINES; ++i) {
		if (header[i].type != NULL) {
			fprintf( f, "%s: %s\n", header[i].type, header[i].value );
		}
	}
	fprintf( f, ".\n" );

	fflush( f );

	if (f != stdout && fclose(f) != 0) {
		fprintf( debugFile, "LIBVOL : can not close `%s' : %s\n", fname, strerror(errno) );
		return 1;
	}

	// Append raw file to vol file
	int fd;

	if (strcmp( fname, "" ) == 0) {
		fd = 1;
	} else {
		fd = open( fname, O_CREAT | O_WRONLY | O_APPEND,  0644 );
		if (fd < 0) {
			fprintf( debugFile, "LIBVOL : can not reopen \"%s\" : %s !", fname, strerror(errno) );
			return 1;
		}
	}
	if (internalDumpRaw( fd ) != 0)
		return 1;

	if (fd != 1 && close( fd ) != 0) {
		fprintf( debugFile, "LIBVOL : can not close `%s' : %s\n", fname, strerror(errno) );
		return 1;
	}
	return 0;
}


int Vol::dumpRaw( const char *fname ) {

	assert( state_ok );

	int fd;
	if (strcmp( fname, "" ) == 0) {
		fd = 1;
	} else {
		fd = open( fname, O_CREAT | O_WRONLY,  0644 );
		if (fd == 0) {
			fprintf( debugFile, "LIBVOL : can not open `%s' : %s !", fname, strerror(errno) );
			return 1;
		}
	}
	internalDumpRaw( fd );

	if (strcmp( fname, "" ) != 0 && close( fd ) != 0) {
		fprintf( debugFile, "LIBVOL : can not close `%s' : %s\n", fname, strerror(errno) );
		return 1;
	}

	return 0;
}


int Vol::internalDumpRaw( int fd ) {

	assert( state_ok );

	ssize_t bytes = 0;

//	Obsolete code from v1
//	bytes += write( fd, &sx, sizeof(int) );
//	bytes += write( fd, &sy, sizeof(int) );
//	bytes += write( fd, &sz, sizeof(int) );
//	bytes += write( fd, "\n", 1 );

	ssize_t curbytes = 0;
	do {
		ssize_t errcode = write( fd, data + curbytes, total*sizeof(voxel) - curbytes );
		if (errcode == -1 && errno != EINTR) {
			fprintf( debugFile, "LIBVOL : Can't write file : %s\n", strerror(errno) );
			return 1;
		} else {
			curbytes += errcode;
		}
	} while ((unsigned)curbytes != total*sizeof(voxel));
	bytes += curbytes;

	// Check all was OK.
	if ((unsigned)bytes != total*sizeof(voxel) /* + 3*sizeof(int) + 1 */) {
		fprintf( debugFile, "LIBVOL : It seems I couldn't write raw file ...\n" );
		return 1;
	}

	return 0;

}


int Vol::getHeaderField( const char *type ) const {

	assert( state_ok );

	for (int i = 0; i < MAX_HEADERNUMLINES; ++i) {
		if (header[i].type != NULL && strcmp( header[i].type, type ) == 0 ) {
			return i;
		}
	}
	return -1;
}


const char *Vol::getHeaderValue( const char *type ) const {

	int i = getHeaderField( type );
	if (i == -1)
		return NULL;
	return header[i].value;

}

int Vol::getHeaderValueAsDouble( const char *type, double *dest ) const {

	assert( state_ok );
	float fdest;
	int ret;

	int i = getHeaderField( type );
	if (i == -1)
		return 1;

	ret = sscanf( header[i].value, "%e", &fdest );
	*dest = fdest;
	return ret == 1;
}


int Vol::getHeaderValueAsInt( const char *type, int *dest ) const {

	assert( state_ok );

	int i = getHeaderField( type );
	if (i == -1)
		return 1;

	return sscanf( header[i].value, "%d", dest ) != 0;
}


int Vol::setHeaderValue( const char *type, const char *value ) {

	assert( state_ok );

	int ind = getHeaderField( type );
	if (ind != -1) {
		header[ind] = HeaderField( type, value );
		return 0;
	}
	int i;
	for (i = 0; i < MAX_HEADERNUMLINES && header[i].type != NULL; ++i) ;
	if (i == MAX_HEADERNUMLINES)
		return 1;
	header[i] = HeaderField( type, value );
	return 0;
}


int Vol::setHeaderValue( const char *type, int value ) {

	assert( state_ok );

	char buf[30];
	snprintf( buf, 29, "%d", value );
	return setHeaderValue( type, buf );

}


int Vol::setHeaderValue( const char *type, double value ) {

	assert( state_ok );

	char buf[30];
	snprintf( buf, 29, "%f", value );
	return setHeaderValue( type, buf );

}

Vol::endian_t Vol::initEndian() {

	Vol::endian_t e;
	e.i_endian.i = 0;
	for (unsigned int i = 0; i < sizeof(e.i_endian.i); ++i) {
		e.i_endian.i += (i + '0') << (i*8);
	}
	e.v_endian.v = 0;
	for (unsigned int i = 0; i < sizeof(e.v_endian.v); ++i) {
		e.v_endian.v += (i + '0') << (i*8);
	}

	e.i_endian.ci[sizeof(int)] = 0;
	e.v_endian.cv[sizeof(voxel)] = 0;

	return e;
}

void Vol::setVolumeCenter( int x, int y, int z ) {

	assert( state_ok );

	cx = x;
	cy = y;
	cz = z;

	setHeaderValue( "Center-X", x );
	setHeaderValue( "Center-Y", y );
	setHeaderValue( "Center-Z", z );
}

void Vol::drawAxis( ) {

	assert( state_ok );

	setHeaderValue( "Axis", 1 );

	int mins[] = {minX(), minY(), minZ()};
	int maxs[] = {maxX(), maxY(), maxZ()};

	for (int j = mins[0]; j < maxs[0]; ++j)
		(*this)(j, cy, cz) = 0x3F;
	for (int j = mins[1]; j < maxs[1]; ++j)
		(*this)(cx, j, cz) = 0x7F;
	for (int j = mins[2]; j < maxs[2]; ++j)
		(*this)(cx, cy, j) = 0xFF;
}

const Vol::endian_t Vol::endian = Vol::initEndian();


bool Vol::inBounds( int x, int y, int z ) const {

	x -= minX();
	y -= minY();
	z -= minZ();

	return x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz;

}

Vol &Vol::operator &= (const Vol &v) {

	if ( v.sx != sx || v.sy != sy || v.sz != sz ) {
		resize( max( sx, v.sx ), max( sy, v.sy ), max( sz, v.sz ) );
	}

    //Center copy
    cx = v.cx;
    cy = v.cy;
    cz = v.cz;

	int alpha = 0, valpha = 0;
	getHeaderValueAsInt( "Alpha-Color", &alpha );
	v.getHeaderValueAsInt( "Alpha-Color", &valpha );

	for (int i = 0; i < v.sx; ++i)
		for (int j = 0; j < v.sy; ++j)
			for (int k = 0; k < v.sz; ++k) {
			  int pos = posOf( i, j, k );
				int vpos = v.posOf( i, j, k ) ;
				if (data[pos] == alpha || v.data[vpos] == valpha) {
					data[pos] = alpha;
				}
			}

	return *this;
}

Vol &Vol::operator |= (const Vol &v) {

	if ( v.sx != sx || v.sy != sy || v.sz != sz ) {
		resize( max( sx, v.sx ), max( sy, v.sy ), max( sz, v.sz ) );
	}
	int px = abs(sx - v.sx)/2;
	int py = abs(sy - v.sy)/2;
	int pz = abs(sz - v.sz)/2;

	fprintf( debugFile, "LIBVOL : or : %d %d %d\n", px, py, pz );

	int alpha = 0, valpha = 0;
	getHeaderValueAsInt( "Alpha-Color", &alpha );
	v.getHeaderValueAsInt( "Alpha-Color", &valpha );

	for (int i = 0; i < v.sx; ++i)
	    for (int j = 0; j < v.sy; ++j)
		    for (int k = 0; k < v.sz; ++k) {
				int pos = posOf( i + px, j + py, k + pz );
				int vpos = v.posOf( i, j, k );
				if (data[pos] == alpha && v.data[vpos] != valpha) {
					data[pos] = (v.data[vpos] == alpha) ? valpha : v.data[vpos];
				}

			}

	return *this;
}


Vol &Vol::operator -= (const Vol &v) {

	if ( v.sx != sx || v.sy != sy || v.sz != sz ) {
		resize( max( sx, v.sx ), max( sy, v.sy ), max( sz, v.sz ) );
	}

	int alpha = 0, valpha = 0;
	getHeaderValueAsInt( "Alpha-Color", &alpha );
	v.getHeaderValueAsInt( "Alpha-Color", &valpha );

	for (int i = 0; i < v.sx; ++i)
		for (int j = 0; j < v.sy; ++j)
			for (int k = 0; k < v.sz; ++k) {
				int pos = posOf( i, j, k ) ;
				int vpos = posOf( i, j, k );
				if (v.data[vpos] != valpha) {
					data[pos] = alpha;
				}
			}

	return *this;
}

voxel Vol::alpha() const {

	int color = 0;
	getHeaderValueAsInt( "Alpha-Color", &color );
	return color;

}

void Vol::resize( int nsx, int nsy, int nsz ) {

	int ntotal = nsx * nsy * nsz;

	if (total == ntotal) // in fact we don't have to resize
		return;

	int px = (nsx - sx)/2, py = (nsy - sy)/2, pz = (nsz - sz)/2;
	int alpha_color = alpha();
	voxel *ndata;

	try {
		ndata = new voxel[ntotal];
	} catch (...) {
		fprintf( debugFile, "LIBVOL : Can not allocate %d KBytes !\n", total/1024 );
		state_ok = false;
		return;
	}

	for (int i = 0; i < total; ++i)
		ndata[i] = alpha_color;

	for (int i = 0; i < sx; ++i) {
		for (int j = 0; j < sy; ++j) {
			for (int k = 0; k < sz; ++k) {
				int pos = posOf( i, j, k );
			//	int npos = (i + (nsx - sx)/2)*nsy*nsx + (j + (nsy - sy)/2)*nsx + k + (nsz - sz)/2;
			//	int npos = posOf( i + (nsx - sx)/2, j + (nsy - sy)/2, k + (nsz - sz)/2 );
				int npos = (i + px)*nsx*nsy + (j + py)*nsx + (k + pz);
				ndata[npos] = data[pos];
			}
		}
	}

	delete []data;
	data = ndata;
	total = ntotal;
	sx = nsx;
	sy = nsy;
	sz = nsz;
	setHeaderValue( "X", sx );
	setHeaderValue( "Y", sy );
	setHeaderValue( "Z", sz );

	fprintf( debugFile, "LIBVOL : resize\n" );
}

bool Vol::rotatePoint( int i, int j, int k, double rx, double ry, double rz, int *inx, int *iny, int *inz) {

	double	crx = cos(rx), srx = sin(rx),
			cry = cos(ry), sry = sin(ry),
			crz = cos(rz), srz = sin(rz);
	double 	m[3][3][3] = {{ 	{1., 0., 0.},
							{0., crx, -srx},
							{0., srx, crx} },
						{	{cry, 0., sry},
							{0., 1., 0.},
							{-sry, 0., cry} },
						{ 	{crz, -srz, 0.},
							{srz, crz, 0.},
							{0., 0., 1.} } };

	double 	im[3][3][3];

	double x = (double)(i - sx/2);
	double y = (double)(j - sy/2);
	double z = (double)(k - sz/2);

	for (int i = 0; i < 3; ++i) {
		invmat33( im[i], m[i] );

		double nx = im[i][0][0]*x + im[i][0][1]*y + im[i][0][2]*z;
		double ny = im[i][1][0]*x + im[i][1][1]*y + im[i][1][2]*z;
		double nz = im[i][2][0]*x + im[i][2][1]*y + im[i][2][2]*z;

		x = nx;
		y = ny;
		z = nz;
	}


	*inx = (int)x + sx/2;
	*iny = (int)y + sy/2;
	*inz = (int)z + sz/2;

	return (*inx >= 0 && *inx < sx && *iny >= 0 && *iny < sy && *inz >= 0 && *inz < sz);

}

void Vol::rotate( double rx, double ry, double rz ) {

	int alpha_color = alpha();

	voxel *ndata;
	try {
		ndata = new voxel[total];
	} catch (...) {
		fprintf( debugFile, "LIBVOL : Can not allocate %d KBytes !\n", total/1024 );
		state_ok = false;
		return;
	}

	for (int i = 0; i < total; ++i)
		ndata[i] = alpha_color;

	for (int i = 0; i < sx; ++i) {
		for (int j = 0; j < sy; ++j) {
			for (int k = 0; k < sz; ++k) {
				int nx, ny, nz;
				if (!rotatePoint( i, j, k, rx, ry, rz, &nx, &ny, &nz ))
					continue;

				int pos = posOf( i, j, k );
				int sourcepos = posOf( nx, ny, nz );

				// FIXME : this should never happen
				if (sourcepos < 0 || sourcepos >= total || pos < 0 || pos >= total) {
					fprintf( debugFile, "LIBVOL : bug : source = %d, pos = %d, "
							"total = %d, sx = %d sy = %d sz = %d, x = %d y = %d, z = %d\n",
							sourcepos, pos, total, sx, sy, sz, i, j, k );
					continue;
				}

				ndata[pos] = data[sourcepos];
			}
		}
	}

	delete []data;
	data = ndata;
}

void Vol::symetry( int maxx, int maxy, int maxz ) {

    int mins[] = {-maxx, -maxy, -maxz};
    int maxs[] = {maxx + 1, maxy + 1, maxz + 1};

    for (int x = mins[0]; x < 0; ++x) {
        for (int y = 0; y < maxs[1]; ++y) {
            for (int z = 0; z < maxs[2]; ++z) {
                (*this)( x, y, z ) = (*this)( -x, y, z );
            }
        }
    }

    for (int x = mins[0]; x < maxs[0]; ++x) {
        for (int y = mins[1]; y < 0; ++y) {
            for (int z = 0; z < maxs[2]; ++z) {
                (*this)( x, y, z ) = (*this)( x, -y, z );
            }
        }
    }

    for (int x = mins[0]; x < maxs[0]; ++x) {
        for (int y = mins[1]; y < maxs[1]; ++y) {
            for (int z = mins[2]; z < 0; ++z) {
                (*this)( x, y, z ) = (*this)( x, y, -z );
            }
        }
    }
}

static void translateInitLoop( int *begin, int *end, int *step, int v, int m ) {

	if (v > 0) {
		*begin = m;
		*end = 0;
		*step = -1;
	} else {
		*begin = 0;
		*end = m;
		*step = 1;
	}

}

void Vol::translate( int vx, int vy, int vz ) {

	int begin[3], end[3], step[3];

	translateInitLoop( begin, end, step, vx, sx );
	translateInitLoop( begin + 1, end + 1, step + 1, vy, sy );
	translateInitLoop( begin + 2, end + 2, step + 2, vz, sz );

	int alpha_color = alpha();

	for (int i = begin[0]; i != end[0]; i += step[0])
		for (int j = begin[1]; j != end[1]; j += step[1])
			for (int k = begin[2]; k != end[2]; k += step[2]) {

				int spos = posOf(i + vx, j + vy, k + vz);
				if (spos >= 0 && spos < total)
				 	data[posOf( i, j, k )] = data[spos];
				else
					data[posOf( i, j, k )] = alpha_color;

			}


}

static int invmat33( double inv[3][3], double mat[3][3] )
{
  double t4, t6, t8, t10, t12, t14, t1;

  t4 = mat[0][0]*mat[1][1];
  t6 = mat[0][0]*mat[1][2];
  t8 = mat[0][1]*mat[1][0];
  t10 = mat[0][2]*mat[1][0];
  t12 = mat[0][1]*mat[2][0];
  t14 = mat[0][2]*mat[2][0];
  t1 = (t4*mat[2][2]-t6*mat[2][1]-t8*mat[2][2]+
        t10*mat[2][1]+t12*mat[1][2]-t14*mat[1][1]);

  if(t1 == 0)
    return 0;

  inv[0][0] = (mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1])/t1;
  inv[0][1] = -(mat[0][1]*mat[2][2]-mat[0][2]*mat[2][1])/t1;
  inv[0][2] = (mat[0][1]*mat[1][2]-mat[0][2]*mat[1][1])/t1;
  inv[1][0] = -(mat[1][0]*mat[2][2]-mat[1][2]*mat[2][0])/t1;
  inv[1][1] = (mat[0][0]*mat[2][2]-t14)/t1;
  inv[1][2] = -(t6-t10)/t1;
  inv[2][0] = (mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0])/t1;
  inv[2][1] = -(mat[0][0]*mat[2][1]-t12)/t1;
  inv[2][2] = (t4-t8)/t1;
  return 1;
}

/*
int Vol::posOf( int x, int y, int z ) const {

	//return x*sx*sy + y*sx + z;
	return z*sx*sy + y*sx + x;
}
*/

#ifdef NDEBUG
FILE* const Vol::debugFile = fopen( "/dev/null", "w" );
#else
FILE* const Vol::debugFile = stderr;
#endif
