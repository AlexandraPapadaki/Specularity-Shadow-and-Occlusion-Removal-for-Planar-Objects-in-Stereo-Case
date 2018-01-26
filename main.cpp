#include "App.h"

int getNumberButton(int key) {
	if (key == BUTTON_ZERO) return 0;
	if (key < BUTTON_ONE || key > BUTTON_ZERO) return -1;
	return key - 48;
}

int main(void)
{
	/*while (1) {
		bool accepted = false;
		int b0 = waitKey(0);
		int num0, num1, res = -1;
		if (-1 != (num0 = getNumberButton(b0))) {
			int b1 = waitKey(1000);
			int num1 = getNumberButton(b1);

			if (b1 == BUTTON_OK) {
				res = num0;
				accepted = true;
			}
			if (num1 != -1) break;
		}
		if (accepted) cout << res << endl;
	}*/

	Mat tar, src;
	App::InitApp(tar, src);
	App::DisplayImage(tar, "target image", -1);
	App::DisplayImage(src, "source image", 500);

	// check if object rectification is desired
	cout << "Do you attempt to rectify the object of interest? (y / n)" << endl;
	bool bRect = 0;
	while (1) {
		char key = getchar();
		if (key == 'y' || key == 'Y') {
			bRect = 1;
			break;
		}
		else if (key == 'n' || key == 'N') {
			bRect = 0;
			break;
		}
	}

	destroyAllWindows();
	// run alignment with rectification if desired
	App::RunAlignment(bRect);

	// check if shadow detection is desired
	bool bShadowDet[2] = { 0, 0 };
	std::string str[2] = { "target", "source" };
	for (int i = 0; i < 2; i++) {
		cout << "Do you attempt to remove shadows from the "
			<< str[i] << " image? (y / n)" << endl;
		while (1) {
			char key = getchar();
			if (key == 'y' || key == 'Y') {
				cout << "Shadow detection will be applied for "
					<< str[i] << " image." << endl << endl;
				bShadowDet[i] = 1;
				break;
			}
			else if (key == 'n' || key == 'N') {
				cout << "Skipping shadow detection for "
					<< str[i] << " image." << endl << endl;
				break;
			}
		}
	}

	destroyAllWindows();

	App::SetupAlgorithm(bShadowDet[0], bShadowDet[1]);

	App::RunShadowDetection();
	// run specularity detection and 
	// correction and get initial result
	destroyAllWindows();
	Mat result = App::ComputeResults();
	App::ConsolePrint("Increase Threshold with +");
	App::ConsolePrint("Decrease Threshold with -");
	//App::ConsolePrint("Reapply Specularity Detection with 1");
	App::ConsolePrint("Reapply Shadow Detection with 2");
	App::ConsolePrint("Save Result with s");

	while (1) {
		App::DisplayImage(tar, "target image", -1);
		App::DisplayImage(src, "source image", -1);
		App::DisplayImage(result, "result", -1);

		// wait for button pressed
		int button = 0;
		while (1) {
			button = waitKey(0);
			if (button == BUTTON_PLUS) {
				result = App::IncreaseThreshold();
				break;
			}
			if (button == BUTTON_MINUS) {
				result = App::DecreaseThreshold();
				break;
			}
			if (button == BUTTON_TWO) {
				App::RunShadowDetection();
				result = App::GetCurrentResult();
				break;
			}
#ifdef EXPERIMENTAL
			if (button == BUTTON_ONE) {
				result = App::ComputeResults();
				break;
			}
			if (button == BUTTON_THREE) {
				App::ShowRegionSimilarities();
			}
#endif
			if (button == BUTTON_SAVE) {
				imwrite("result.jpg", result);
			}
		}

	}


}
