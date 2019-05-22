/******************************************************************************/
/*                                                                            */
/*        Licensed Materials - Property of IBM                                */
/*                                                                            */
/*        IBM Power Vector Intrinisic Functions V1.0.1                        */
/*                                                                            */
/*        Copyright IBM Corp. 2015                                            */
/*        US Government Users Restricted Rights - Use, duplication or         */
/*        disclosure restricted by GSA ADP Schedule Contract with IBM Corp.   */
/*                                                                            */
/*        THIS IS FOR IBM INTERNAL USE ONLY, NOT TO BE RELEASED !             */
/*                                                                            */
/******************************************************************************/

#ifndef _H_PMMINTRIN
#define _H_PMMINTRIN

#include                      "vec128sp.h"

#define  _mm_lddqu_si128      vec_load1qu
#define  _mm_hadd_ps          vec_partialhorizontal2sp

#endif //_H_PMMINTRIN
