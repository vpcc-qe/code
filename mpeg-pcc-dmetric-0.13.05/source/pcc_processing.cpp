/*
 * Software License Agreement
 *
 *  Point to plane metric for point cloud distortion measurement
 *  Copyright (c) 2017, MERL
 *
 *  All rights reserved.
 *
 *  Contributors:
 *    Dong Tian <tian@merl.com>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory.h>
#include <sys/stat.h>

#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
#include <map>

#include "pcc_processing.hpp"

#if _WIN32
#define stat64 _stat64
#endif

using namespace std;
using namespace pcc_processing;

// trim from start
static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
  return ltrim(rtrim(s));
}

/**!
 * **************************************
 *  Class PointXYZSet
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

PointXYZSet::~PointXYZSet()
{
}

void
PointXYZSet::init( long int size, int i0, int i1, int i2 )
{
  if (i0 >= 0)
    idxInLine[0] = i0;
  if (i1 >= 0)
    idxInLine[1] = i1;
  if (i2 >= 0)
    idxInLine[2] = i2;

  p.resize( size );
  nbdup.resize( size );
  std::fill(nbdup.begin(), nbdup.end(), 1);
  original_indices.resize(size);
  for (size_t i = 0; i < size; i++) {
      original_indices[i] = i;
  }
}

/**!
 * \brief
 *  loadPoints
 *
 * \param out
 *  return 0, if succeed
 *  return non-zero, otherwise
 *
 *  Dong Tian <tian@merl.com>
 */
int
PointXYZSet::loadPoints( PccPointCloud *pPcc, long int idx )
{
  if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT32 )
  {
    for (int i = 0; i < 3; i++)
      p[idx][i] = *(float*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT64 )
  {
    for (int i = 0; i < 3; i++)
      p[idx][i] = *(double*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::INT32 )
  {
    for (int i = 0; i < 3; i++)
      p[idx][i] = *(int32_t*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::UINT32 )
  {
    for (int i = 0; i < 3; i++)
      p[idx][i] = *(uint32_t*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else
  {
    cerr << "Error! Wrong xyz data type: " << pPcc->fieldType[ idxInLine[0] ] << endl;
    return -1;
  }
  return 0;
}

/**!
 * **************************************
 *  Class RGBSet
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

/**!
 * \brief
 *  RGBSet
 *
 *  Dong Tian <tian@merl.com>
 */
RGBSet::~RGBSet()
{
}

void
RGBSet::init( long int size, int i0, int i1, int i2 )
{
  if (i0 >= 0)
    idxInLine[0] = i0;
  if (i1 >= 0)
    idxInLine[1] = i1;
  if (i2 >= 0)
    idxInLine[2] = i2;

  c.resize( size );
}

int
RGBSet::loadPoints( PccPointCloud *pPcc, long int idx )
{
  if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::UINT8 )
  {
    for (int i = 0; i < 3; i++)
      c[idx][i] = *(unsigned char*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else
  {
    cerr << "Error! Wrong rgb data type: " << pPcc->fieldType[ idxInLine[0] ] << endl;
    return -1;
  }
  return 0;
}

/**!
 * **************************************
 *  Class NormalSet
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

NormalSet::~NormalSet()
{
}

void
NormalSet::init( long int size, int i0, int i1, int i2 )
{
  if (i0 >= 0)
    idxInLine[0] = i0;
  if (i1 >= 0)
    idxInLine[1] = i1;
  if (i2 >= 0)
    idxInLine[2] = i2;

  n.resize( size );
}

/**!
 * \brief
 *   loadPoints
 *
 * \param out
 *   return 0, if succeed
 *   return non-zero, otherwise
 *
 *  Dong Tian <tian@merl.com>
 */
int
NormalSet::loadPoints( PccPointCloud *pPcc, long int idx )
{
  if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT32 )
  {
    for (int i = 0; i < 3; i++)
      n[idx][i] = *(float*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT64 )
  {
    for (int i = 0; i < 3; i++)
      n[idx][i] = *(double*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else
  {
    cerr << "Error! Wrong xyz data type: " << pPcc->fieldType[ idxInLine[0] ] << endl;
    return -1;
  }
  return 0;
}

/**!
 * **************************************
 *  Class LidarSet
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

LidarSet::~LidarSet()
{
}

void
LidarSet::init( long int size, int i0 )
{
  if (i0 >= 0)
    idxInLine[0] = i0;

  reflectance.resize( size );
}

int
LidarSet::loadPoints( PccPointCloud *pPcc, long int idx )
{
  if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::UINT16 )
  {
    reflectance[ idx ] = *(unsigned short*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[0] ]);
  }
  else if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::UINT8 )
  {
    reflectance[ idx ] = *(unsigned char*)(pPcc->lineMem.get() + pPcc->fieldPos[ idxInLine[0] ]);
  }
  else
  {
    cerr << "Error! Wrong reflectance data type: " << pPcc->fieldType[ idxInLine[0] ] << endl;
    return -1;
  }
  return 0;
}

/**!
 * **************************************
 *  Class PccPointCloud
 *    To load point clouds
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

PccPointCloud::PccPointCloud()
{
  size = 0;
  fileFormat = -1;
  fieldNum = 0;
  dataPos = 0;
  lineNum = 0;

  bXyz = bRgb = bNormal = bLidar = false;
}

PccPointCloud::~PccPointCloud()
{
}

/**!
 * \brief
 *  checkFile
 *
 * \param out
 * return the position of the field, if found
 * return -1 if not found
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::checkFile(string fileName)
{
  ifstream in;
  string line;
  string str[3];
  int counter = 0;

  int secIdx = 0;               // 1: vertex

  bool bPly = false;
  bool bEnd = false;
  fieldSize = 0;                  // position in the line memory buffer

  in.open(fileName, ifstream::in);
  while ( getline(in, line) && lineNum < 100) // Maximum number of lines of header section: 100
  {
    line = rtrim( line );
    if (line == "ply")
    {
      bPly = true;
    }
    else
    {
      counter = 0;
      stringstream ssin(line);
      while (ssin.good() && counter < 3){
        ssin >> str[counter];
        counter++;
      }

      if ( str[0] == "format" )
      {
        if ( str[1] == "ascii" )
          fileFormat = 0;
        else if ( str[1] == "binary_little_endian" )
          fileFormat = 1;
        else
        {
          cerr << "Format not to be handled: " << line << endl;
          return -1;
        }
      }
      else if ( str[0] == "element" )
      {
        if (str[1] == "vertex")
        {
          size = stol( str[2] );
          secIdx = 1;
        }
        else
          secIdx = 2;
      }
      else if ( str[0] == "property" && secIdx == 1 && str[1] != "list" )
      {
        fieldPos.push_back(fieldSize);
        if ( str[1] == "uint8" || str[1] == "uchar" ) {
          fieldType.push_back(PointFieldTypes::UINT8);
          fieldSize += sizeof(unsigned char);
        }
        else if ( str[1] == "int8" || str[1] == "char" ) {
          fieldType.push_back(PointFieldTypes::INT16);
          fieldSize += sizeof(char);
        }
        else if ( str[1] == "uint16" || str[1] == "ushort" ) {
          fieldType.push_back(PointFieldTypes::UINT16);
          fieldSize += sizeof(unsigned short);
        }
        else if ( str[1] == "int16" || str[1] == "short" ) {
          fieldType.push_back(PointFieldTypes::INT16);
          fieldSize += sizeof(short);
        }
        else if ( str[1] == "uint" || str[1] == "uint32" ) {
          fieldType.push_back(PointFieldTypes::UINT32);
          fieldSize += sizeof(unsigned int);
        }
        else if ( str[1] == "int" || str[1] == "int32" ) {
          fieldType.push_back(PointFieldTypes::INT32);
          fieldSize += sizeof(int);
        }
        else if ( str[1] == "float" || str[1] == "float32" ) {
          fieldType.push_back(PointFieldTypes::FLOAT32);
          fieldSize += sizeof(float);
        }
        else if ( str[1] == "float64" || str[1] == "double" ) {
          fieldType.push_back(PointFieldTypes::FLOAT64);
          fieldSize += sizeof(double);
        }
        else {
          cerr << "Incompatible header section. Line: " << line << endl;
          bPly = false;
        }
        fieldNum++;
      }
      else if ( str[0] == "end_header" )
      {
        lineNum++;
        bEnd = true;
        break;
      }
    }

    lineNum++;
  }
#ifndef _WIN32
  dataPos = in.tellg(); // Only to be used under Linux. @DT
#endif

  in.close();

  lineMem.reset(new unsigned char[fieldNum * sizeof(double)]);

  if (bPly && fileFormat >= 0 && fieldNum > 0 && size > 0 && bEnd)
    return 0;

  cout << "Warning: The header section seems not fully compatible!\n";
  return -1;
}

/*
 * \brief
 *   Check the attribute by parsing the header section
 *
 * \param[in]
 *   fileName: Input point cloud file
 *   fieldName: Attribute name
 *   fieldTypeX: Potential data type of the attribute
 *
 * \return -1, if fails
 *         index of this attribute in a row, if succeeds
 *
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::checkField(string fileName, string fieldName, const std::initializer_list<const char*>& fieldTypes)
{
  ifstream in;
  string line;
  string str[3];
  int counter = 0;
  bool bFound = false;
  int iPos = -1;

  int secIdx = 0;               // 1: vertex.
  in.open(fileName, ifstream::in);

  while ( getline(in, line) )
  {
    if (line == "ply")
      ;
    else
    {
      // cout << line << endl;
      counter = 0;
      stringstream ssin(line);
      while (ssin.good() && counter < 3){
        ssin >> str[counter];
        counter++;
      }

      if ( str[0] == "format" )
        ;                       // do nothing
      else if ( str[0] == "element" )
      {
        if (str[1] == "vertex")
          secIdx = 1;
        else
          secIdx = 2;
      }
      else if ( str[0] == "property" && secIdx == 1 && str[1] != "list" )
      {
        iPos++;
        if ( std::find(fieldTypes.begin(), fieldTypes.end(), str[1]) != fieldTypes.end() )
        {
          if ( str[2] == fieldName )
          {
            bFound = true;
            break;
          }
        }
      }
    }
  }
  in.close();

  if ( !bFound )
    iPos = -1;

  return iPos;
}

/*
 * \brief
 *   Load one line of data from the input file
 *
 * \param[in]
 *   in: Input file stream
 *
 * \return 0, if succeed
 *         non-zero, if failed
 *
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::loadLine( ifstream &in )
{
  int i;
  if (fileFormat == 0)          // ascii
  {
    for (i = 0; i < fieldNum; i++)
    {
      switch (fieldType[i])
      {
      case PointFieldTypes::UINT8:
        in >> ((unsigned short*)(lineMem.get()+fieldPos[i]))[0]; // @DT: tricky. not using uint8 when reading
        break;
      case PointFieldTypes::INT8:
        in >> ((short*)(lineMem.get()+fieldPos[i]))[0];
        break;
      case PointFieldTypes::UINT16:
        in >> ((unsigned short*)(lineMem.get()+fieldPos[i]))[0];
        break;
      case PointFieldTypes::INT16:
        in >> ((short*)(lineMem.get()+fieldPos[i]))[0];
        break;
      case PointFieldTypes::UINT32:
        in >> ((unsigned int*)(lineMem.get()+fieldPos[i]))[0];
        break;
      case PointFieldTypes::INT32:
        in >> ((int*)(lineMem.get()+fieldPos[i]))[0];
        break;
      case PointFieldTypes::FLOAT32:
        in >> ((float*)(lineMem.get()+fieldPos[i]))[0];
        break;
      case PointFieldTypes::FLOAT64:
        in >> ((double*)(lineMem.get()+fieldPos[i]))[0];
        break;
      default:
        cout << "Unknown field type: " << fieldType[i] << endl;
        return -1;
        break;
      }
    }
  }
  else if (fileFormat == 1)     // binary_little_endian
  {
    for (i = 0; i < fieldNum; i++)
    {
      switch (fieldType[i])
      {
      case PointFieldTypes::UINT8:
        in.read((char*)(lineMem.get()+fieldPos[i]), sizeof(unsigned char));
        break;
      case PointFieldTypes::INT8:
        in.read((char*)(lineMem.get()+fieldPos[i]), sizeof(char));
        break;
      case PointFieldTypes::UINT16:
        in.read((char*)(lineMem.get()+fieldPos[i]), sizeof(unsigned short));
        break;
      case PointFieldTypes::INT16:
        in.read((char*)(lineMem.get()+fieldPos[i]), sizeof(short));
        break;
      case PointFieldTypes::UINT32:
        in.read((char*)(lineMem.get()+fieldPos[i]), sizeof(unsigned int));
        break;
      case PointFieldTypes::INT32:
        in.read((char*)(lineMem.get()+fieldPos[i]), sizeof(int));
        break;
      case PointFieldTypes::FLOAT32:
        in.read((char*)(lineMem.get()+fieldPos[i]), sizeof(float));
        break;
      case PointFieldTypes::FLOAT64:
        in.read((char*)(lineMem.get()+fieldPos[i]), sizeof(double));
        break;
      default:
        cout << "Unknown field type: " << fieldType[i] << endl;
        return -1;
        break;
      }
    }
  }
  else
    return -1;

  return 0;
}

#if _WIN32
int
PccPointCloud::seekAscii(ifstream &in)
{
  string line;
  int lineNum = 0;
  bool bFound = false;
  while (!bFound && lineNum < 1000 && getline(in, line))
  {
    line = rtrim(line);
    if (line == "end_header")
      bFound = true;
  }
  if (bFound)
    return 0;
  return -1;
}

int
PccPointCloud::seekBinary(ifstream &in)
{
  // Move the pointer to the beginning of the data
  // in.seekg( dataPos );
  char buf[200];
  char mark[] = "end_header";
  int  markLen = sizeof(mark);
  int posFile = 0;
  int posBuf = 0;
  while (strcmp(buf, mark) != 0 && posFile < 9000)
  {
    in.read(&buf[posBuf], 1);
    buf[posBuf + 1] = 0;
    if (posBuf >= markLen - 1)
    {
      memcpy(buf, buf + 1, markLen + 1);
      buf[markLen - 1] = 0;
    }
    else
    {
      posBuf++;
    }
    posFile++;
  }
  // Move pointer to after end of line for this line
  buf[0] = 0;
  buf[1] = 0;
  while (strcmp(buf, "\n") != 0 && posFile < 9000)
  {
    in.read(&buf[0], 1);
    posFile++;
  }

  if (posFile >= 9000)
    return -1;
  return 0;
}
#endif


/*
 * \brief load data into memory
 * \param[in]
 *   inFile: File name of the input point cloud
 *   normalsOnly: Import only normals, or all data
 *   dropDuplicates: 0 (detect only), 1 (drop), 2 (average)
 *
 * \return 0, if succeed
 *         non-zero, if failed
 *
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::load(string inFile, bool normalsOnly, int dropDuplicates, int neighborsProc)
{
  int ret = 0;

  int iPos[3];                  // The position of x,y,z in the list

  if (inFile == "")
    return 0;

  struct stat64 bufferExist;
  if (stat64(inFile.c_str(), &bufferExist) != 0)
  {
    cout << "File does not exist: " << inFile << endl;
    return -1;
  }

  // Check file header
  if ( checkFile( inFile ) != 0 )
    return -1;

  // Make sure (x,y,z) available and determine the float type
  iPos[0] = checkField( inFile, "x", {"float", "float32", "float64", "double", "int32", "uint32"} );
  iPos[1] = checkField( inFile, "y", {"float", "float32", "float64", "double", "int32", "uint32"} );
  iPos[2] = checkField( inFile, "z", {"float", "float32", "float64", "double", "int32", "uint32"} );
  if ( iPos[0] < 0 && iPos[1] < 0 && iPos[2] < 0 )
    return -1;
  xyz.init(size, iPos[0], iPos[1], iPos[2]);
  bXyz = true;

  // Optional normals (nx,ny,nz)
  iPos[0] = checkField( inFile, "nx", {"float", "float32", "float64", "double", "int32", "uint32"} );
  iPos[1] = checkField( inFile, "ny", {"float", "float32", "float64", "double", "int32", "uint32"} );
  iPos[2] = checkField( inFile, "nz", {"float", "float32", "float64", "double", "int32", "uint32"} );
  if ( iPos[0] >= 0 && iPos[1] >= 0 && iPos[2] >= 0 ) {
    normal.init(size, iPos[0], iPos[1], iPos[2]);
    bNormal = true;
  }

  if (normalsOnly && !bNormal)
    return -1;

  // Make sure (r,g,b) available and determine the float type
  iPos[0] = checkField( inFile, "red",   {"uint8", "uchar"} );
  iPos[1] = checkField( inFile, "green", {"uint8", "uchar"} );
  iPos[2] = checkField( inFile, "blue",  {"uint8", "uchar"} );
  if ( !normalsOnly && iPos[0] >= 0 && iPos[1] >= 0 && iPos[2] >= 0 )
  {
    rgb.init(size, iPos[0], iPos[1], iPos[2]);
    bRgb = true;
  }

  // Make sure (lidar) available and determine the integer type
  iPos[0] = checkField( inFile, "reflectance", {"uint16", "uint8"} );
  if ( !normalsOnly && iPos[0] < 0 )
    iPos[0] = checkField( inFile, "refc", {"uint16", "uint8"} );

  if ( iPos[0] >= 0 )
  {
    lidar.init(size, iPos[0]);
    bLidar = true;
  }

  // Load regular data (rather than normals)
  ifstream in;

  if (fileFormat == 0)
  {
    in.open(inFile, ifstream::in);
#if _WIN32
    if (seekAscii(in) != 0)
    {
      cout << "Check the file header section. Incompatible header!" << endl;
      in.close();
      return -1;
    }
#else
    in.seekg(dataPos);
#endif
  }
  else if (fileFormat == 1)
  {
    in.open(inFile, ifstream::in | ifstream::binary);
#if _WIN32
    if (seekBinary(in) != 0)
    {
      cout << "Check the file header section. Incompatible header!" << endl;
      in.close();
      return -1;
    }
#else
    in.seekg(dataPos);
#endif
  }

  for (long int i = 0; i < size; i++)
  {
    if ( loadLine( in ) < 0 )   // Load the data into line memory
    {
      ret = -1;
      break;
    }
    xyz.loadPoints( this, i );
    if (bNormal)
      normal.loadPoints( this, i );
    if (bRgb)
      rgb.loadPoints( this, i );
    if (bLidar)
      lidar.loadPoints( this, i );
  }
  in.close();

#if 1
  if (1)
  {
    long int i = size - 1;
    cout << "Verifying if the data is loaded correctly.. The last point is: ";
    cout << xyz.p[i][0] << " " << xyz.p[i][1] << " " << xyz.p[i][2] << endl;
  }
#endif

  // sort the point cloud
  // std::sort(this->begin(), this->end());
  std::vector<size_t> indices(size);
  for (size_t i = 0; i < size; i++) {
      indices[i] = i;
  }

  std::sort(indices.begin(), indices.end(),
      [this](size_t i1, size_t i2) {
          return this->xyz.p[i1] < this->xyz.p[i2];
      });

  std::vector<PointXYZSet::point_type> temp_p = xyz.p;
  std::vector<int> temp_nbdup = xyz.nbdup;
  std::vector<size_t> temp_orig = xyz.original_indices;

  for (size_t i = 0; i < size; i++) {
      size_t sorted_idx = indices[i];
      xyz.p[i] = temp_p[sorted_idx];
      xyz.nbdup[i] = temp_nbdup[sorted_idx];
      xyz.original_indices[i] = sorted_idx; 

      if (bNormal)
          normal.n[i] = normal.n[sorted_idx];
      if (bRgb)
          rgb.c[i] = rgb.c[sorted_idx];
      if (bLidar)
          lidar.reflectance[i] = lidar.reflectance[sorted_idx];
  }
  // Find runs of identical point positions
  for (auto it_seq = this->begin(); it_seq != this->end(); ) {
    it_seq = std::adjacent_find(it_seq, this->end());
    if (it_seq == this->end())
      break;

    // accumulators for averaging attribute values
    long cattr[3] {}; // sum colors.
    long lattr {}; // sum lidar.

    // iterate over the duplicate points, accumulating values
    int count = 0;
    auto it = it_seq;
    for (; *it == *it_seq; ++it) {
      count++;
      int idx = it.idx;
      if (bRgb) {
        cattr[0] += rgb.c[idx][0];
        cattr[1] += rgb.c[idx][1];
        cattr[2] += rgb.c[idx][2];
      }

      if (bLidar)
        lattr += lidar.reflectance[idx];
    }

    int first_idx = it_seq.idx;
    it_seq = it;

    // averaging case only
    if (dropDuplicates != 2)
      continue;

    if (bRgb) {
      rgb.c[first_idx][0] = cattr[0] / count;
      rgb.c[first_idx][1] = cattr[1] / count;
      rgb.c[first_idx][2] = cattr[2] / count;
    }

    if (neighborsProc == 2)
      xyz.nbdup[first_idx] = count;

    if (bLidar)
      lidar.reflectance[first_idx] = lattr / count;
  }

  int duplicatesFound = 0;
  if (dropDuplicates != 0) {
    auto last = std::unique(this->begin(), this->end());
    duplicatesFound = std::distance(last, this->end());

    size -= duplicatesFound;
    xyz.p.resize(size);
    xyz.nbdup.resize(size);
    if (bNormal)
      normal.n.resize(size);
    if (bRgb)
      rgb.c.resize(size);
    if (bLidar)
      lidar.reflectance.resize(size);
  }

  if (duplicatesFound > 0)
  {
    switch (dropDuplicates)
    {
    case 0:
      printf("WARNING: %d points with same coordinates found\n",
              duplicatesFound);
      break;
    case 1:
      printf("WARNING: %d points with same coordinates found and dropped\n",
              duplicatesFound);
      break;
    case 2:
      printf("WARNING: %d points with same coordinates found and averaged\n",
              duplicatesFound);
    }
  }

#if 0
  {
    ofstream outF;
    outF.open( "testItPly.ply", ofstream::out );
    for (long int i = 0; i < size; i++)
    {
      outF << xyz.p[i][0] << " " << xyz.p[i][1] << " " << xyz.p[i][2] << " " << (unsigned short) rgb.c[i][0] << " " << (unsigned short) rgb.c[i][1] << " " << (unsigned short) rgb.c[i][2] << " " << lidar.reflectance[i] << endl;
      // outF << xyz.p[i][0] << " " << xyz.p[i][1] << " " << xyz.p[i][2] << " " << (unsigned short) rgb.c[i][0] << " " << (unsigned short) rgb.c[i][1] << " " << (unsigned short) rgb.c[i][2] << endl;
      // outF << xyz.p[i][0] << " " << xyz.p[i][1] << " " << xyz.p[i][2] << endl;
    }
    outF.close();
  }
#endif

  return ret;
}
