import { getSortedPosts } from "@utils/content-utils";
import { getPostUrlBySlug } from "@utils/url-utils";

export async function GET() {
	const posts = await getSortedPosts();
	const blogs = posts.map((post) => ({
		title: post.data.title,
		description: post.data.description,
		link: new URL(getPostUrlBySlug(post.slug), import.meta.env.SITE).href,
	}));

	return new Response(
		JSON.stringify({
			total: blogs.length,
			blogs,
		}),
		{
			headers: {
				"Content-Type": "application/json; charset=utf-8",
			},
		},
	);
}
