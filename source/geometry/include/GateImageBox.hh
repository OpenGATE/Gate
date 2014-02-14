/**
 * \class       GateImageBox
 * \brief       Displays an image with OpenGL.
 * \details     Behaves like a G4Box for tracking but displays image with three orthogonal planes in OpenGL: axial, sagittal and coronal.
 * \author      Jérôme Suhard <jerome@suhard.fr>
 * \date        02/2014
 * \warning     Makes the assumption that OpenGL headers are still accessible.
 * \warning     Only works in Immediate mode ( /vis/open OGLI ).
 * \copyright   GNU Lesser General Public Licence (LGPL).
 */

#ifndef GATEIMAGEBOX_HH
#define GATEIMAGEBOX_HH

#ifdef G4VIS_USE_OPENGL
#define G4VIS_BUILD_OPENGL_DRIVER 1
#ifdef G4VIS_USE_OPENGLX
#define G4VIS_BUILD_OPENGLX_DRIVER 1
#endif
#ifdef G4VIS_USE_OPENGLQT
#define G4VIS_BUILD_OPENGLQT_DRIVER 1
#endif
#include "G4OpenGLSceneHandler.hh"
#include "G4OpenGLStoredSceneHandler.hh"
#include "G4OpenGL.hh"
#endif

#include "G4Box.hh"

#include "GateImage.hh"

class GateImageBox : public virtual G4Box {
    
public:
    GateImageBox(const GateImage & image, const G4String & name);
    ~GateImageBox();
    
    void DescribeYourselfTo(G4VGraphicsScene& scene) const;
private:
#ifdef G4VIS_USE_OPENGL
    void DescribeYourselfTo(G4OpenGLSceneHandler& scene) const;
    
    std::vector<GateImage::PixelType> getXYSlice(const GateImage & image, const size_t z) const;
    std::vector<GateImage::PixelType> getXZSlice(const GateImage & image, const size_t y) const;
    std::vector<GateImage::PixelType> getYZSlice(const GateImage & image, const size_t x) const;
    GLubyte * convertToRGB(std::vector<GateImage::PixelType> slice, GateImage::PixelType min, GateImage::PixelType max) const;
    GLuint genOpenGLTexture(const GLubyte * rgb, int width, int height) const;
    void initOpenGLTextures(const GateImage & image, const size_t x, const size_t y, const size_t z);
    
    size_t position_x;
    size_t position_y;
    size_t position_z;
    size_t resolution_x;
    size_t resolution_y;
    size_t resolution_z;
    GLuint texture_xy;
    GLuint texture_xz;
    GLuint texture_yz;
#endif
};

#endif /* GATEIMAGEBOX_HH */
