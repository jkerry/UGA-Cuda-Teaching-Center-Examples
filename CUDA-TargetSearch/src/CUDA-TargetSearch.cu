/*
 * main.cpp
 *
 *  Created on: Jul 25, 2012
 *      Author: john
 */
#include <tclap/CmdLine.h>
#include <string>
#include <stdlib.h>
#include <time.h>

#include "bitmap_image.hpp"
#include "kernel_wrapper.h"

#define RES 256
#define R_BG 100
#define G_BG 100
#define B_BG 100

bool targetPositionsNotAcceptable(int numTargets, int numPix ,int* targets) {
	for(int i = 0; i < numTargets; i++){
		//first, check if the index is within the last 2 possible positions
		if(targets[i] == (numPix - 1) || targets[i] == (numPix - 2))
			return true;

		for(int j = 0; j < numTargets; j++){ //compare ith target with all others
			if(j == i) continue;//dont compare a target with itself

			if( targets[i] >= targets[j] && targets[i] <= (targets[j]+2))
				return true; //if the target index is withing the 3 cell range about another, we have a false generation
		}
	}
	return false;
}

int main(int argc, char** argv) {

	try {
		//create the command flag parser
		TCLAP::CmdLine cmd(
				"CUDA-TargetSearch, a program for quickly analyzing two images, identifying all possible suspect pixel groups, and highlighting identical matches in the two images.",
				' ', "1.0");
		//define a flag argument to switch image generation vs image analysis
		TCLAP::SwitchArg generate("g", "generate", "generate test images",
				false);
		//define a value for the first image name
		TCLAP::ValueArg<std::string> firstImage("f", "first",
				"The first image to generate/analyze.", false, "first.bmp",
				"string");
		//define a value for the second image name
		TCLAP::ValueArg<std::string> secondImage("s", "second",
				"The second image to generate/analyze.", false, "second.bmp",
				"string");
		//define a value for the number of potential targets in each image
		TCLAP::ValueArg<int> numTargets("t", "targets",
				"The number of targets to generate in both images", false, 10,
				"int");

		//add the arguments to the parser
		cmd.add(generate);
		cmd.add(firstImage);
		cmd.add(secondImage);
		cmd.add(numTargets);

		//parse the arguments
		cmd.parse(argc, argv);

		bool gen = generate.getValue();
		std::string first = firstImage.getValue();
		std::string second = secondImage.getValue();
		int targets = numTargets.getValue();

		if (gen) {
			std::cout << "Generating images: " << first << "  " << second
					<< " with " << targets << " targets." << std::endl;
			/*
			 *	Common operations
			 */
			//seed the random number generator
			srand(time(NULL));
			//allocate array for the potential target positions
			int* targetPositions2 = new int[targets];
			int* targetPositions1 = new int[targets];
			int rt; //temporary variable

			bitmap_image image1(RES, RES);
			image1.clear();
			//allocate color arrays for image1
			unsigned char* r1 = new unsigned char[image1.pixel_count()];
			unsigned char* g1 = new unsigned char[image1.pixel_count()];
			unsigned char* b1 = new unsigned char[image1.pixel_count()];
			//set background to predefined color
			for (unsigned int i = 0; i < image1.pixel_count(); ++i) {
				r1[i] = R_BG;
				g1[i] = G_BG;
				b1[i] = B_BG;
			}
			//Randomly select positions for the targets
			do {
				for (int i = 0; i < targets; i++) {
					targetPositions1[i] = (int) ((float) rand()
							/ (float) RAND_MAX * (float) image1.pixel_count());
				}
			} while (targetPositionsNotAcceptable(targets, image1.pixel_count() ,targetPositions1));

			//add targets to those positions
			//for each target, add a white, color, white pattern
			for (int i = 0; i < targets; i++) {
				rt = targetPositions1[i];
				r1[rt] = 255;
				r1[rt + 1] = rand() % 255;
				r1[rt + 2] = 255;
				g1[rt] = 255;
				g1[rt + 1] = rand() % 255;
				g1[rt + 2] = 255;
				b1[rt] = 255;
				b1[rt + 1] = 7 * (1 + i) % 255; //fixed incline in blue values to insure unique targets between images
				b1[rt + 2] = 255;
			}



			/*
			 * Second image operations
			 */
			//generate the first image using the pre-processor defined square resolution RESxRES
			bitmap_image image2(RES, RES);
			image2.clear();
			//allocate color arrays for image2
			unsigned char* r2 = new unsigned char[image2.pixel_count()];
			unsigned char* g2 = new unsigned char[image2.pixel_count()];
			unsigned char* b2 = new unsigned char[image2.pixel_count()];
			//set background to predefined color
			for (unsigned int i = 0; i < image2.pixel_count(); ++i) {
				r2[i] = R_BG;
				g2[i] = G_BG;
				b2[i] = B_BG;
			}
			//Randomly select positions for the targets
			do {
				for (int i = 0; i < targets; i++) {
					targetPositions2[i] = (int) ((float) rand()
							/ (float) RAND_MAX * (float) image2.pixel_count());
				}
			} while (targetPositionsNotAcceptable(targets, image2.pixel_count() ,targetPositions2));

			//add targets to those positions
			//for each target, add a white, color, white pattern
			for (int i = 0; i < targets; i++) {
				rt = targetPositions2[i];
				r2[rt] = 255;
				r2[rt + 1] = rand() % 255;
				r2[rt + 2] = 255;
				g2[rt] = 255;
				g2[rt + 1] = rand() % 255;
				g2[rt + 2] = 255;
				b2[rt] = 255;
				b2[rt + 1] = 7 * (targets + 1 + i) % 255; //fixed incline in blue values to insure unique targets between images
				b2[rt + 2] = 255;
			}


			/*
			 * Verify that the blue channel doesnt match for any pair of target centers
			 */
			for(int i = 0; i < targets; i++){
				unsigned char image1_tColor = b1[targetPositions1[i]+1];
				for(int j = 0; j < targets; j++){
					unsigned char image2_tColor = b2[targetPositions1[i]+1];
					if(image1_tColor == image2_tColor)std::cout<<"Match Encountered! Generate Again!"<<std::endl;
				}
			}

			/*
			 * Place the target of interest by matching the colored pattern
			 */

			int p1 = targetPositions1[0];
			int p2 = targetPositions2[0];

			//set image1 to white/red/white pattern
			r1[p1+1] = 255;
			g1[p1+1] = 0;
			b1[p1+1] = 0;
			//set image 2 to white/red/white pattern
			r2[p2+1] = 255;
			g2[p2+1] = 0;
			b2[p2+1] = 0;

			/*
			 * Output the images
			 */
			image1.import_rgb(r1, g1, b1);
			image2.import_rgb(r2, g2, b2);
			image1.save_image(first);
			image2.save_image(second);

		} else {
			std::cout << "Processing images: " << first << "  " << second
					<< std::endl;
			bitmap_image image1(first);
			bitmap_image image2(second);

			unsigned char* r1 = new unsigned char[image1.pixel_count()];
			unsigned char* g1 = new unsigned char[image1.pixel_count()];
			unsigned char* b1 = new unsigned char[image1.pixel_count()];

			unsigned char* r2 = new unsigned char[image2.pixel_count()];
			unsigned char* g2 = new unsigned char[image2.pixel_count()];
			unsigned char* b2 = new unsigned char[image2.pixel_count()];
			
			image1.export_rgb(r1,g1,b1);
			image2.export_rgb(r2,g2,b2);

			std::cout<<"Exit code: "<<kernelLaunchpad(r1, g1, b1, r2, g2, b2, image1.pixel_count(), image2.pixel_count())<<std::endl;
		}

	} catch (TCLAP::ArgException &e) // catch any exceptions
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
				<< std::endl;
	}
	return 0;
}

