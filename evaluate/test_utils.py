import mutinfo.distributions.images.geometric as geometric
import mutinfo.distributions.images.field as field

def get_image_generation_fn(drawing_fn_name, coord_fn_name, image_shape, min_size, max_size=None):
    

    if coord_fn_name == "symmetric_gaussian_field":
        drawing_fn = getattr(field, drawing_fn_name)

        def generate_image(x,y):
            return (
                drawing_fn(x, field.symmetric_gaussian_field, image_shape),
                drawing_fn(y, field.symmetric_gaussian_field, image_shape)
            )
        return generate_image
    
    drawing_fn = getattr(geometric, drawing_fn_name)
    coord_fn = getattr(geometric, coord_fn_name)
    def generate_image(x, y):
        return (
        drawing_fn(coord_fn(x, min_size=(0.2, 0.2)), image_shape),
        drawing_fn(coord_fn(y, min_size=(0.2, 0.2)), image_shape)
        )
    return generate_image