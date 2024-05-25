## TW-CUI-Util
This repository contains multiple 'utility' nodes for various ease-of-use improvements and combining a number of nodes necessary for a workflow into less.

---

### "Generation" nodes
The following "generation" helper nodes are provided:

 - **Common SDXL Resolutions** - A simple node that allows selection of known common SDXL resolutions that work well for SDXL model generations.
 - **Generation Parameters** - A node that combines a number of requisite things such as CFG scale, image dimensions, checkpoint selection, VAE
   selection, scheduler selection, and other things necessary for KSampler and others to work into one node. This node ALSO produces requisite MODEL,
   VAE, CLIP, SCHEDULER, and SAMPLER items which can be passed to KSampler or CLIP encoding for prompts.
 - **Prompts** - One of the requisites for most SDXL and other image generation is "text to image" which requires prompts to be provided. This node provides
   the capacity to define both positive and negative prompts, produces those as strings, but also produces the POSITIVE and NEGATIVE CONDITIONING items to pass to KSampler.

### "Image" nodes
The following "image" helper nodes are provided:

 - **Save Image** - Modeled after the in-built ComfyUI Save Image node, this node has limited configuration options beond the filename prefix format, time format, and the
   ability to determine if we include ComfyUI default metadata in the output image as embedded EXIF.  It only saves as PNGs, and does not have advanced metadata other than
   the contents of the workflow added to it like ComfyUI does by default.  There is a more advanced version of this that lets you pass in all sorts of data and get metadata
   in easier formats (like Automatic1111 does).
 - **Save Image (Advanced)** - This is based somewhat on [a preexisting "comfy-image-saver"](https://github.com/giriss/comfy-image-saver) repository, but with a different
   implementation of saving and the collection of a lot more metadata than the base "Save Image" node.  It defaults to a lower compression level (5 out of 9) in order to be
   faster at saving images, in the way ComfyUI's inbuilt nodes do. It also provides metadata in the Parameters field in Automatic1111 format of easy-to-read parameters.  It
   also has the capacity to save the entire workflow from ComfyUI into the metadata as well for easy replication of the generation in ComfyUI directly.

   This node also has much more advanced capabilities for filename patterns, subdirectory paths underneath the base "outputs" directory, and other nifty components.

### "Util" nodes
The following basic "utility" nodes are provided, which don't exist in standard ComfyUI:

 - **BOOLEAN Literal** - A node that lets you define a BOOLEAN (true/false) value to pass into other nodes.
 - **FLOAT Literal** - A node that lets you define a FLOAT value to pass into other nodes.
 - **INTEGER Literal** - A node that lets you define an INTEGER value to pass into other nodes.
 - **STRING Literal** - A node that lets you define a non-multiline STRING value to pass into other nodes.
 - **STRING Literal (Multiline)** - A node that lets you define a multiline STRING value to pass into other nodes.
