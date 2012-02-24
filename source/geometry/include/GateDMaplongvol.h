/*
Copyright 2002, 2003 Alexis Guillaume <aguillau@liris.univ-lyon2.fr> for "Laboratoire LIRIS, université Lyon II, France."
Copyright 2002, 2003 David Coeurjolly <dcoeurjo@liris.univ-lyon2.fr> for "Laboratoire LIRIS, université Lyon II, France."

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
#ifndef LIBLONGVOL
#define LIBLONGVOL

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//! Voxel type
typedef long lvoxel;

//! This class handles lvoxel 3D pictures. It can save them to disk and read them.
//! See test.cc for an example of use of this class.
class Longvol {
public:
	//! Constructor to create 'from scratch' a new volume.
	//! No file is created.
	Longvol( int sx, int sy, int sz, lvoxel tcolor );

	//! Constructor to build a volume from a .vol file.
	//! If filename is "", stdin is used.
	Longvol( const char *fname );

	//! Constructor to build a volume from a .raw file *without* size informations.
	Longvol( const char *fname, int sx, int sy, int sz, lvoxel defaultAlpha = 0 );
	
	//! Default constructor
	Longvol();

	//! Destructor (not virtual, so don't inherit this class !)
	~Longvol();

 lvoxel * getDataPointer() { return data; }

	//! Copy constructor
	Longvol( const Longvol & );
	//! Assignement operator
	const Longvol &operator = (const Longvol &);

	//! Operator to access a pixel
	/*! Usage : mylongvolume(0, 0, 0) = mycolor; */
	lvoxel &	operator()(int x, int y, int z);
	//! Operateur to access a pixel when object is const
	lvoxel 	operator()(int x, int y, int z) const;

	//! Some people could prefer this syntax to get/set a lvoxel
	lvoxel 		get( int x, int y, int z ) const { return (*this)(x, y, z); }
	void		set( int x, int y, int z, lvoxel c ) { (*this)(x, y, z) = c; }

	//! Write this longvolume object into a .longvol file.
	//! If filename is "", stdout is used.
	//! Return > 0 if error.
	int 		dumpLongvol( const char *filename );
	//! Write this volume object into a .raw file
	//! If filanem is "", stdout is used.
	//! Return > 0 if error
	int			dumpRaw( const char *filename );


	//! Returns the state of the object (false : bad, true : ok)
	bool 	isOK() { 
		return state_ok;
	}

	//! vol union 
	Longvol &operator |= (const Longvol &);
	//! longvol and
	Longvol &operator &= (const Longvol &);
	//! longvol minus
	Longvol &operator -= (const Longvol &v);

	
	//! Returns NULL if this field is not found
	const char *getHeaderValue( const char *type ) const;
	//! Returns non-zero if failure
	int			getHeaderValueAsDouble( const char *type, double *dest ) const;
	//! Returns non-zero if failure
	int			getHeaderValueAsInt( const char *type, int *dest ) const;
	
	//! Returns non-zero on failure
	int setHeaderValue( const char *type, const char *value );
	//! Returns non-zero on failure
	int setHeaderValue( const char *type, lvoxel value );
	//! Returns non-zero on failure
	int setHeaderValue( const char *type, int value );
	//! Returns non-zero on failure
	int setHeaderValue( const char *type, double value );

	//! You can set the coordinates of the center of the longvolume.
	//! By default, center is (0, 0, 0).
	void	setVolumeCenter( int x, int y, int z );

	//! Draw Axis X, Y and Z
	void		drawAxis( );

	
	//! longvol invert (/!\ All colors are lost)
	void		invert();
	//! rotate image in longvolume
	void		rotate( double rx, double ry, double rz );
	//! will compute the symetry of all positives points in the longvolume.
	void		symetry( int maxx, int maxy, int maxz ); 
	//! Translate contents of the longvolume
	//! New lvoxels are transparents
	void		translate( int vx, int vy, int vz );


	//! X-size of the 3D longvolume
	int		sizeX() const { return sx; }
	//! Y-size
	int		sizeY() const { return sy; }
	//! Z-sizw
	int		sizeZ() const { return sz; }

	//! The least x
	int 	minX() const { return cx - sx/2; }
	//! The least y
	int		minY() const { return cy - sy/2; }
	//! The least z
	int 	minZ() const { return cz - sz/2; }

	//! The greatest X ( warning : it is not included in bounds )
	int		maxX() const { return cx + sx/2 + sx%2; }
	//! The greatest Y ( warning : it is not included in bounds )
	int		maxY() const { return cy + sy/2 + sy%2; }
	//! The greatest Z ( warning : it is not included in bounds )
	int		maxZ() const { return cz + sz/2 + sz%2; }
	
	//! The X-coordinate of the center
	int 	cX() const { return cx; }
	//! The Y-coordinate of the center
	int		cY() const { return cy; }
	//! The Z-coordinates of the center
	int 	cZ() const { return cz; }
	

	//! Returns true if the point is valid for this longvolume
	bool	inBounds( int x, int y, int z ) const;

	//! Returns alpha color
	lvoxel 	alpha() const;
	
	//! Position of a point in the lvoxel array
	inline
	int		posOf( int x, int y, int z ) const {
		return z*sx*sy + y*sx + x;
	}

private:
	//! Internal method for cpying
	void copy( const Longvol & );
	//! Internal method for destroying
	void destroy();
	//! Internal resize method
	void resize( int nsx, int nsy, int nsz );

	//! The 3D matrix of lvoxel
	lvoxel 		*data;
	//! Dimensions of the matrix
	int			sx, sy, sz;
	//! Number of lvoxels (equals sx*sy*sz)
	int			total;

	//! Read longvol data from a file already open
	int		readLongvolData( FILE *in );
	//! Read raw data from a file already open 
	int		readV1RawData( FILE *in, bool headerInited, lvoxel defaultAlpha );
	//! Read raw data from a file already open
	int		readV2RawData( FILE *in, bool headerInited, int sx, int sy, int sz, lvoxel defaultAlpha );
	
	//! Write .raw data into fd
	int		internalDumpRaw( int fd );
	

	bool	rotatePoint( int, int, int, double, double, double, int *, int *, int * );

	//! This class help us to associate a field type and his value.
	//! An object is a pair (type, value). You can copy and assign
	//! such objects.
	/*! In recent C++, we should use a std::map, but we prefer (badly) code it 
		by hand for compatibility with old compilers.
		At this time, there is a limit of 30 fields in header :-} */
	struct HeaderField {
		//! Constructor. The string are copied.
		HeaderField( const char *t, const char *v ) :
			type( strdup(t) ), value( strdup(v) ) {}
		~HeaderField() {
			free( type );
			free( value );
		}
		//! Copy constructor
		HeaderField( const HeaderField &h ) :
			type( strdup(h.type) ), value( strdup(h.value) ) {};
		//! Default constructor
		HeaderField() : type(NULL), value(NULL) {};	
		//! Assignement operator
		const HeaderField &operator = (const HeaderField &h) {
			free( type );
			free( value );
			if (h.type != NULL) {
				type = strdup( h.type );
				value = strdup( h.value );
			}
			return *this;
		}
		//! Type of field (e.g. Voxel-Size)
		char *type;
		//! Value of field (e.g. 2)
		char *value;
	};

	//! Maximum number of fields in a .longvol file header
	static const int MAX_HEADERNUMLINES = 64;

	//! Here are the fields. A field is empty when his type string is NULL.
	HeaderField header[ MAX_HEADERNUMLINES ];
	
	//! Internal method which returns the index of a field or -1 if not found.
	int getHeaderField( const char *type ) const;

	//! Global list of required fields in a .longvol file
	static const char *requiredHeaders[];

	//! Global state of this object1
	bool state_ok;

	//! A little structure that help us to determine host endian
	struct endian_t {
		//! Endian for the int type
		union {
			int 	i;
			char 	ci[ sizeof(int) + 1 ];
		} i_endian;
		
		//! Endian for the voxel type
		union {
			lvoxel	v;
			char 	cv[ sizeof(lvoxel) + 1 ];
		} v_endian;
	};
	static const endian_t endian;
	static endian_t initEndian();

	int cx, cy, cz;
	
	static 	FILE *const debugFile;
	
};

inline
Longvol operator&( Longvol v1, const Longvol &v2 ) {
	v1 &= v2;
	return v1;
}

inline
Longvol operator|( Longvol v1, const Longvol &v2 ) {
	v1 |= v2;
	return v1;
}

inline
Longvol operator-( Longvol v1, const Longvol &v2 ) {
	v1 -= v2;
	return v1;
}

#endif
